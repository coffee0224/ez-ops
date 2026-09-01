"""TileLang backend for the fused_o_proj_ffn op.

Port of mega-qwen's ``ldg_o_proj_postnorm_mlp`` decode phase as persistent
kernels: every stage launches exactly num_sms CTAs that pull row-tile work
units from a flat list, the TileLang equivalent of the source kernel's
persistent grid. TileLang has no grid-wide barrier, so the source kernel's
AtomicGridSync phase boundaries become four back-to-back launches inside one
module (like tilelang's own split flash-decode example):

  1. o_proj + residual     x2 = residual + attn @ Wo.T
  2. norm + gate/up GEMV   xn = rmsnorm(x2) * post_norm_w; g/u = xn @ W.T
  3. SwiGLU                act = silu(g) * u   (the source's g_mlp_intermediate)
  4. down + residual       out = x2 + act @ Wd.T

GEMV recipe (decode is M=1, tl-style gemm needs M>=16): each unit stages the
full input row plus one weight tile in shared, multiplies into a fragment,
and cross-thread-reduces along K. Gate and up are separate work units sharing
one weight buffer and one pipeline (the source streams gate_row/up_row
together; the split keeps the pipeline inside the 99 KiB smem opt-in cap).
Intermediates (x2, g, u, act) live in fp32 module globals, matching the
source kernel's float workspaces — slightly closer to the fp32 oracle than
the bf16-staged reference, within the op's documented tolerance.

The per-unit RMS statistic is recomputed redundantly (every unit re-reads the
2-8 KB x2 row), trading negligible L2 traffic for no extra sync.
"""

import logging

import tilelang
import torch
from tilelang import language as T
from tilelang.carver.arch.driver.cuda_driver import get_num_sms

logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)

from ..base_kernel import BaseKernel
from ...registry import register_kernel

RMS_EPS = 1e-6


@register_kernel("fused_o_proj_ffn", "tilelang")
class FusedOProjFfnTileLangKernel(BaseKernel):
    _block_n = 64  # output rows per work unit
    _block_k = 128  # K-tile of every GEMV pipeline
    _num_stages = 2
    _threads = 128

    def __init__(
        self,
        batch: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
    ):
        self.batch = batch
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_rows = num_heads * head_dim
        if hidden_size & (hidden_size - 1):
            raise ValueError(f"hidden_size ({hidden_size}) must be a power of two")
        if hidden_size % self._block_k or intermediate_size % self._block_k or self.q_rows % self._block_k:
            raise ValueError(
                f"hidden_size ({hidden_size}), intermediate_size ({intermediate_size}) and "
                f"num_heads * head_dim ({self.q_rows}) must be divisible by BLOCK_K={self._block_k}"
            )
        self.num_sms = get_num_sms()
        self._kernel = None

    def _make_kernel(self):
        B, H, I, Q = self.batch, self.hidden_size, self.intermediate_size, self.q_rows
        BN, BK = self._block_n, self._block_k
        NSTAGES, THREADS = self._num_stages, self._threads
        eps, num_sms = RMS_EPS, self.num_sms

        @tilelang.jit(out_idx=[7])
        def jit_fn():
            @T.prim_func
            def main(
                A: T.Buffer((B, Q), "bfloat16"),
                R: T.Buffer((B, H), "bfloat16"),
                Wo: T.Buffer((H, Q), "bfloat16"),
                NW: T.Buffer((H,), "bfloat16"),
                WG: T.Buffer((I, H), "bfloat16"),
                WU: T.Buffer((I, H), "bfloat16"),
                Wd: T.Buffer((H, I), "bfloat16"),
                Out: T.Buffer((B, H), "bfloat16"),
            ):
                # fp32 workspaces (the source kernel's float buffers); bf16
                # globals trip tilelang's storage-legalizer on this version.
                X2 = T.alloc_global((B, H), "float32")
                G = T.alloc_global((B * I,), "float32")
                U = T.alloc_global((B * I,), "float32")
                Act = T.alloc_global((B * I,), "float32")

                # -- stage 1: x2 = residual + attn @ Wo.T --------------------
                with T.Kernel(num_sms, threads=THREADS) as bx:
                    As = T.alloc_shared((Q,), "bfloat16")
                    Ws = T.alloc_shared((BN, BK), "bfloat16")
                    prod = T.alloc_fragment((BN, BK), "float32")
                    part = T.alloc_fragment((BN,), "float32")
                    acc = T.alloc_fragment((BN,), "float32")

                    num_tiles = H // BN
                    total = B * num_tiles
                    num_iters = (total + num_sms - 1) // num_sms
                    for it in T.serial(num_iters):
                        wid = it * num_sms + bx
                        if wid < total:
                            b = wid // num_tiles
                            row0 = (wid % num_tiles) * BN
                            for j in T.Parallel(Q):
                                As[j] = A[b, j]
                            T.fill(acc, 0)
                            for k in T.Pipelined(Q // BK, num_stages=NSTAGES):
                                T.copy(Wo[row0 : row0 + BN, k * BK : (k + 1) * BK], Ws)
                                for i, j in T.Parallel(BN, BK):
                                    prod[i, j] = Ws[i, j].astype("float32") * As[k * BK + j].astype("float32")
                                T.reduce_sum(prod, part, dim=1, clear=True)
                                for i in T.Parallel(BN):
                                    acc[i] += part[i]
                            for i in T.Parallel(BN):
                                X2[b, row0 + i] = acc[i] + R[b, row0 + i].astype("float32")

                # -- stage 2: xn = rmsnorm(x2) * NW; g/u = xn @ W.T -----------
                with T.Kernel(num_sms, threads=THREADS) as bx:
                    Ws = T.alloc_shared((BN, BK), "bfloat16")
                    Xs = T.alloc_shared((H,), "float32")
                    NWs = T.alloc_shared((H,), "bfloat16")
                    sq = T.alloc_fragment((H,), "float32")
                    sum_out = T.alloc_fragment((1,), "float32")
                    prod = T.alloc_fragment((BN, BK), "float32")
                    part = T.alloc_fragment((BN,), "float32")
                    acc = T.alloc_fragment((BN,), "float32")

                    num_tiles = I // BN
                    total = B * 2 * num_tiles  # gate units first, then up units
                    num_iters = (total + num_sms - 1) // num_sms
                    for it in T.serial(num_iters):
                        wid = it * num_sms + bx
                        if wid < total:
                            b = wid // (2 * num_tiles)
                            rem = wid % (2 * num_tiles)
                            is_up = rem // num_tiles
                            row0 = (rem % num_tiles) * BN
                            for j in T.Parallel(H):
                                Xs[j] = X2[b, j]
                                NWs[j] = NW[j]
                            for j in T.Parallel(H):
                                sq[j] = Xs[j] * Xs[j]
                            T.reduce_sum(sq, sum_out, dim=0, clear=True)
                            rstd = T.rsqrt(sum_out[0] / H + eps)
                            T.fill(acc, 0)
                            for k in T.Pipelined(H // BK, num_stages=NSTAGES):
                                if is_up == 0:
                                    T.copy(WG[row0 : row0 + BN, k * BK : (k + 1) * BK], Ws)
                                else:
                                    T.copy(WU[row0 : row0 + BN, k * BK : (k + 1) * BK], Ws)
                                for i, j in T.Parallel(BN, BK):
                                    prod[i, j] = Ws[i, j].astype("float32") * (
                                        Xs[k * BK + j] * rstd * NWs[k * BK + j].astype("float32")
                                    )
                                T.reduce_sum(prod, part, dim=1, clear=True)
                                for i in T.Parallel(BN):
                                    acc[i] += part[i]
                            if is_up == 0:
                                for i in T.Parallel(BN):
                                    G[b * I + row0 + i] = acc[i]
                            else:
                                for i in T.Parallel(BN):
                                    U[b * I + row0 + i] = acc[i]

                # -- stage 3: act = silu(g) * u -------------------------------
                with T.Kernel(num_sms, threads=THREADS) as bx:
                    tx = T.get_thread_binding(0)
                    total = B * I
                    stride = num_sms * THREADS
                    num_iters = (total + stride - 1) // stride
                    for it in T.serial(num_iters):
                        idx = it * stride + bx * THREADS + tx
                        if idx < total:
                            g = G[idx]
                            Act[idx] = g / (1.0 + T.exp(-g)) * U[idx]

                # -- stage 4: out = x2 + act @ Wd.T ---------------------------
                with T.Kernel(num_sms, threads=THREADS) as bx:
                    Ws = T.alloc_shared((BN, BK), "bfloat16")
                    Acts = T.alloc_shared((I,), "float32")
                    prod = T.alloc_fragment((BN, BK), "float32")
                    part = T.alloc_fragment((BN,), "float32")
                    acc = T.alloc_fragment((BN,), "float32")

                    num_tiles = H // BN
                    total = B * num_tiles
                    num_iters = (total + num_sms - 1) // num_sms
                    for it in T.serial(num_iters):
                        wid = it * num_sms + bx
                        if wid < total:
                            b = wid // num_tiles
                            row0 = (wid % num_tiles) * BN
                            for j in T.Parallel(I):
                                Acts[j] = Act[b * I + j]
                            T.fill(acc, 0)
                            for k in T.Pipelined(I // BK, num_stages=NSTAGES):
                                T.copy(Wd[row0 : row0 + BN, k * BK : (k + 1) * BK], Ws)
                                for i, j in T.Parallel(BN, BK):
                                    prod[i, j] = Ws[i, j].astype("float32") * Acts[k * BK + j]
                                T.reduce_sum(prod, part, dim=1, clear=True)
                                for i in T.Parallel(BN):
                                    acc[i] += part[i]
                            for i in T.Parallel(BN):
                                Out[b, row0 + i] = (acc[i] + X2[b, row0 + i]).astype("bfloat16")

            return main

        return jit_fn()

    def __call__(
        self,
        attn: torch.Tensor,
        residual: torch.Tensor,
        Wo: torch.Tensor,
        post_norm_w: torch.Tensor,
        Wg: torch.Tensor,
        Wu: torch.Tensor,
        Wd: torch.Tensor,
    ) -> torch.Tensor:
        B, H, I = self.batch, self.hidden_size, self.intermediate_size
        assert attn.is_cuda and attn.dtype == torch.bfloat16
        assert attn.shape == (B, self.q_rows) and attn.is_contiguous()
        assert residual.shape == (B, H) and residual.is_contiguous()
        assert Wo.shape == (H, self.q_rows) and Wo.is_contiguous()
        assert post_norm_w.shape == (H,) and post_norm_w.is_contiguous()
        assert Wg.shape == (I, H) and Wg.is_contiguous()
        assert Wu.shape == (I, H) and Wu.is_contiguous()
        assert Wd.shape == (H, I) and Wd.is_contiguous()

        if self._kernel is None:
            self._kernel = self._make_kernel()
        return self._kernel(attn, residual, Wo, post_norm_w, Wg, Wu, Wd)
