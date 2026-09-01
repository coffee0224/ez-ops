"""Single-launch TileLang backend for the fused_o_proj_ffn op.

Primary TileLang backend: everything in ONE launch with grid=num_sms,
phases separated by software grid barriers — the same shape as the source
kernel's persistent grid + AtomicGridSync. (The stage-split variant is kept
as ``tilelang_multilaunch`` for comparison.)

  1. o_proj + residual     x2 = residual + attn @ Wo.T
  2. norm + gate/up GEMV   xn = rmsnorm(x2) * post_norm_w; g/u = xn @ W.T
  3. SwiGLU                act = silu(g) * u
  4. down + residual       out = x2 + act @ Wd.T

Why the barriers are hand-rolled atomics instead of ``T.sync_grid()``:
``T.sync_grid()`` lowers to ``cooperative_groups::this_grid().sync()``, but
tilelang 0.1.13 does not model it as a memory clobber, so the compiler
happily hoists a later phase's global loads above it — producer->barrier->
consumer through global memory miscompiles (verified: consumer reads ~40%
wrong while the produced buffer itself is correct). The software barrier
(``atomic_add`` arrival + acquire-spin on a generation counter) consists of
real atomic memory operations the scheduler cannot reorder, and orders
correctly. Cost is ~3 us per barrier (x3), replacing the multi-launch
version's kernel-gap overhead.

Barrier state: the wrapper zeroes ``[Counter, Sense]`` before every launch
(one tiny memset, stream-ordered), so each kernel starts from a deterministic
(0, 0) and per-thread generations are the constants 0, 1, 2. A monotonic
cross-launch Sense scheme was tried and deadlocks: the generation bootstrap
load can be sunk below the first barrier by the scheduler, making barrier N
wait for barrier N+1's release.

Workspaces (x2/g/u fp32, act bf16) are passed as kernel arguments instead of
``T.alloc_global``: bf16 globals trip the storage-legalizer, and in-kernel
argument buffers let the wrapper own their lifetime. GEMV tiles are BK=64 so
the three pipelined weight buffers (one per GEMV phase; a shared buffer may
not feed two Pipelined loops) fit the ~99 KiB opt-in smem cap next to the
row staging. Input vectors (attn row, norm weights) are read straight from
global inside the fragment loops — they are never written in-kernel, so the
barrier-ordering hazard does not apply.
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
class FusedOProjFfnPersistentTileLangKernel(BaseKernel):
    _block_n = 64
    _block_k = 64
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
        self._ws = None  # (x2, g, u, act, counter, sense), allocated lazily once

    def _make_kernel(self):
        B, H, I, Q = self.batch, self.hidden_size, self.intermediate_size, self.q_rows
        BN, BK = self._block_n, self._block_k
        NSTAGES, THREADS = self._num_stages, self._threads
        eps, num_sms = RMS_EPS, self.num_sms
        TOTAL_THREADS = num_sms * THREADS

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
                X2: T.Buffer((B, H), "float32"),
                G: T.Buffer((B * I,), "float32"),
                U: T.Buffer((B * I,), "float32"),
                Act: T.Buffer((B * I,), "bfloat16"),
                Counter: T.Buffer((1,), "int32"),
                Sense: T.Buffer((1,), "int32"),
            ):
                with T.Kernel(num_sms, threads=THREADS) as bx:
                    Ws1 = T.alloc_shared((BN, BK), "bfloat16")
                    Ws2 = T.alloc_shared((BN, BK), "bfloat16")
                    Ws4 = T.alloc_shared((BN, BK), "bfloat16")
                    Xs = T.alloc_shared((H,), "bfloat16")
                    Acts = T.alloc_shared((I,), "bfloat16")
                    prod1 = T.alloc_fragment((BN, BK), "float32")
                    part1 = T.alloc_fragment((BN,), "float32")
                    acc1 = T.alloc_fragment((BN,), "float32")
                    sq2 = T.alloc_fragment((H,), "float32")
                    sum2 = T.alloc_fragment((1,), "float32")
                    prod2 = T.alloc_fragment((BN, BK), "float32")
                    part2 = T.alloc_fragment((BN,), "float32")
                    acc2 = T.alloc_fragment((BN,), "float32")
                    prod4 = T.alloc_fragment((BN, BK), "float32")
                    part4 = T.alloc_fragment((BN,), "float32")
                    acc4 = T.alloc_fragment((BN,), "float32")
                    my_gen = T.alloc_local((1,), "int32")
                    my_gen[0] = 0

                    # -- phase 1: x2 = residual + attn @ Wo.T ----------------
                    num_tiles1 = H // BN
                    total1 = B * num_tiles1
                    for it in T.serial((total1 + num_sms - 1) // num_sms):
                        wid = it * num_sms + bx
                        if wid < total1:
                            b = wid // num_tiles1
                            row0 = (wid % num_tiles1) * BN
                            T.fill(acc1, 0)
                            for k in T.Pipelined(Q // BK, num_stages=NSTAGES):
                                T.copy(Wo[row0 : row0 + BN, k * BK : (k + 1) * BK], Ws1)
                                for i, j in T.Parallel(BN, BK):
                                    prod1[i, j] = Ws1[i, j].astype("float32") * A[b, k * BK + j].astype("float32")
                                T.reduce_sum(prod1, part1, dim=1, clear=True)
                                for i in T.Parallel(BN):
                                    acc1[i] += part1[i]
                            for i in T.Parallel(BN):
                                X2[b, row0 + i] = acc1[i] + R[b, row0 + i].astype("float32")

                    # software grid barrier (see module docstring)
                    prev1 = T.atomic_add(Counter[0], 1, "acq_rel", True)
                    if prev1 == TOTAL_THREADS - 1:
                        T.atomic_store(Counter[0], 0, "release")
                        T.atomic_add(Sense[0], 1, "acq_rel")
                    else:
                        with T.While(T.atomic_load(Sense[0], "acquire") <= my_gen[0]):
                            T.evaluate(0)
                    my_gen[0] = my_gen[0] + 1

                    # -- phase 2: xn = rmsnorm(x2)*NW; g/u = xn @ W.T --------
                    num_tiles2 = I // BN
                    total2 = B * 2 * num_tiles2
                    for it in T.serial((total2 + num_sms - 1) // num_sms):
                        wid = it * num_sms + bx
                        if wid < total2:
                            b = wid // (2 * num_tiles2)
                            rem = wid % (2 * num_tiles2)
                            is_up = rem // num_tiles2
                            row0 = (rem % num_tiles2) * BN
                            for j in T.Parallel(H):
                                Xs[j] = X2[b, j].astype("bfloat16")
                            for j in T.Parallel(H):
                                sq2[j] = Xs[j].astype("float32") * Xs[j].astype("float32")
                            T.reduce_sum(sq2, sum2, dim=0, clear=True)
                            rstd = T.rsqrt(sum2[0] / H + eps)
                            T.fill(acc2, 0)
                            for k in T.Pipelined(H // BK, num_stages=NSTAGES):
                                if is_up == 0:
                                    T.copy(WG[row0 : row0 + BN, k * BK : (k + 1) * BK], Ws2)
                                else:
                                    T.copy(WU[row0 : row0 + BN, k * BK : (k + 1) * BK], Ws2)
                                for i, j in T.Parallel(BN, BK):
                                    prod2[i, j] = Ws2[i, j].astype("float32") * (
                                        Xs[k * BK + j].astype("float32") * rstd * NW[k * BK + j].astype("float32")
                                    )
                                T.reduce_sum(prod2, part2, dim=1, clear=True)
                                for i in T.Parallel(BN):
                                    acc2[i] += part2[i]
                            if is_up == 0:
                                for i in T.Parallel(BN):
                                    G[b * I + row0 + i] = acc2[i]
                            else:
                                for i in T.Parallel(BN):
                                    U[b * I + row0 + i] = acc2[i]

                    prev2 = T.atomic_add(Counter[0], 1, "acq_rel", True)
                    if prev2 == TOTAL_THREADS - 1:
                        T.atomic_store(Counter[0], 0, "release")
                        T.atomic_add(Sense[0], 1, "acq_rel")
                    else:
                        with T.While(T.atomic_load(Sense[0], "acquire") <= my_gen[0]):
                            T.evaluate(0)
                    my_gen[0] = my_gen[0] + 1

                    # -- phase 3: act = silu(g) * u --------------------------
                    tx = T.get_thread_binding(0)
                    total3 = B * I
                    stride = num_sms * THREADS
                    for it in T.serial((total3 + stride - 1) // stride):
                        idx = it * stride + bx * THREADS + tx
                        if idx < total3:
                            g = G[idx]
                            Act[idx] = (g / (1.0 + T.exp(-g)) * U[idx]).astype("bfloat16")

                    prev3 = T.atomic_add(Counter[0], 1, "acq_rel", True)
                    if prev3 == TOTAL_THREADS - 1:
                        T.atomic_store(Counter[0], 0, "release")
                        T.atomic_add(Sense[0], 1, "acq_rel")
                    else:
                        with T.While(T.atomic_load(Sense[0], "acquire") <= my_gen[0]):
                            T.evaluate(0)
                    my_gen[0] = my_gen[0] + 1

                    # -- phase 4: out = x2 + act @ Wd.T ----------------------
                    num_tiles4 = H // BN
                    total4 = B * num_tiles4
                    for it in T.serial((total4 + num_sms - 1) // num_sms):
                        wid = it * num_sms + bx
                        if wid < total4:
                            b = wid // num_tiles4
                            row0 = (wid % num_tiles4) * BN
                            for j in T.Parallel(I):
                                Acts[j] = Act[b * I + j]
                            T.fill(acc4, 0)
                            for k in T.Pipelined(I // BK, num_stages=NSTAGES):
                                T.copy(Wd[row0 : row0 + BN, k * BK : (k + 1) * BK], Ws4)
                                for i, j in T.Parallel(BN, BK):
                                    prod4[i, j] = Ws4[i, j].astype("float32") * Acts[k * BK + j].astype("float32")
                                T.reduce_sum(prod4, part4, dim=1, clear=True)
                                for i in T.Parallel(BN):
                                    acc4[i] += part4[i]
                            for i in T.Parallel(BN):
                                Out[b, row0 + i] = (acc4[i] + X2[b, row0 + i]).astype("bfloat16")

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
            dev = attn.device
            self._ws = (
                torch.empty(B, H, device=dev, dtype=torch.float32),
                torch.empty(B * I, device=dev, dtype=torch.float32),
                torch.empty(B * I, device=dev, dtype=torch.float32),
                torch.empty(B * I, device=dev, dtype=torch.bfloat16),
                torch.zeros(2, device=dev, dtype=torch.int32),  # [Counter, Sense]
            )
        x2, g, u, act, state = self._ws
        # Deterministic barrier state for this launch (stream-ordered memset).
        state.zero_()
        counter, sense = state[:1], state[1:]
        return self._kernel(attn, residual, Wo, post_norm_w, Wg, Wu, Wd, x2, g, u, act, counter, sense)
