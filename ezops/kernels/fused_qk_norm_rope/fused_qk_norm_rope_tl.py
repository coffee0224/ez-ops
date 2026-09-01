"""TileLang backend for the fused_qk_norm_rope op.

Port of mega-qwen's ``ldg_qk_norm_rope_cache`` decode phase as a persistent
kernel: the grid is fixed to num_sms CTAs and each CTA pulls (batch, head)
work units from a flat list, mirroring the source kernel's persistent-block
head distribution. The whole stage is one launch.

Per head: load to shared (fp32), RMS statistics via a fragment reduction,
normalize with the per-dim weight, then half-split RoPE by reading the
partner element (+/- D/2) straight out of shared — the shared-memory
equivalent of the source kernel's ``__shfl_sync`` exchange. KV heads also
write the rotated k and the raw v into the cache slot being decoded.
"""

import logging

import tilelang
import torch
from tilelang import language as T
from tilelang.carver.arch.driver.cuda_driver import get_num_sms

# tilelang's KernelCache warns "consider using @tilelang.jit" on cache hits,
# even when we already use @tilelang.jit. Suppress the misleading warning.
logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)

from ..base_kernel import BaseKernel
from ...registry import register_kernel

RMS_EPS = 1e-6


@register_kernel("fused_qk_norm_rope", "tilelang")
class FusedQkNormRopeTileLangKernel(BaseKernel):
    _threads = 128  # one head is D elements; D <= threads maps 1:1

    def __init__(
        self,
        batch: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        position: int,
        num_kv_heads: int | None = None,
    ):
        if head_dim & (head_dim - 1):
            raise ValueError(f"head_dim ({head_dim}) must be a power of two")
        if not 0 <= position < max_seq_len:
            raise ValueError(f"position ({position}) must be in [0, {max_seq_len})")
        self.batch = batch
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.position = position
        self.num_sms = get_num_sms()
        self._kernel = None

    def _make_kernel(self):
        B, NQ, NKV, D = self.batch, self.num_heads, self.num_kv_heads, self.head_dim
        S, pos = self.max_seq_len, self.position
        eps, num_sms, threads = RMS_EPS, self.num_sms, self._threads

        @tilelang.jit(out_idx=[9, 10])
        def jit_fn():
            @T.prim_func
            def main(
                Q: T.Buffer((B, NQ, D), "bfloat16"),
                K: T.Buffer((B, NKV, D), "bfloat16"),
                V: T.Buffer((B, NKV, D), "bfloat16"),
                QW: T.Buffer((D,), "bfloat16"),
                KW: T.Buffer((D,), "bfloat16"),
                COS: T.Buffer((S, D), "bfloat16"),
                SIN: T.Buffer((S, D), "bfloat16"),
                KC: T.Buffer((B, NKV, S, D), "bfloat16"),
                VC: T.Buffer((B, NKV, S, D), "bfloat16"),
                OQ: T.Buffer((B, NQ, D), "bfloat16"),
                OK: T.Buffer((B, NKV, D), "bfloat16"),
            ):
                with T.Kernel(num_sms, threads=threads) as bx:
                    Xs = T.alloc_shared((D,), "float32")
                    sq = T.alloc_fragment((D,), "float32")
                    sum_out = T.alloc_fragment((1,), "float32")

                    total = B * (NQ + NKV)
                    num_iters = (total + num_sms - 1) // num_sms
                    for it in T.serial(num_iters):
                        wid = it * num_sms + bx
                        if wid < total:
                            b = wid // (NQ + NKV)
                            h = wid % (NQ + NKV)
                            if h < NQ:
                                for i in T.Parallel(D):
                                    Xs[i] = Q[b, h, i].astype("float32")
                                for i in T.Parallel(D):
                                    sq[i] = Xs[i] * Xs[i]
                                T.reduce_sum(sq, sum_out, dim=0, clear=True)
                                rstd_q = T.rsqrt(sum_out[0] / D + eps)
                                for i in T.Parallel(D):
                                    Xs[i] = Xs[i] * rstd_q * QW[i].astype("float32")
                                for i in T.Parallel(D):
                                    pair = T.if_then_else(i < D // 2, i + D // 2, i - D // 2)
                                    xp = Xs[pair]
                                    cos = COS[pos, i].astype("float32")
                                    sin = SIN[pos, i].astype("float32")
                                    out = T.if_then_else(
                                        i < D // 2,
                                        Xs[i] * cos - xp * sin,
                                        Xs[i] * cos + xp * sin,
                                    )
                                    OQ[b, h, i] = out.astype("bfloat16")
                            else:
                                kh = h - NQ
                                for i in T.Parallel(D):
                                    Xs[i] = K[b, kh, i].astype("float32")
                                for i in T.Parallel(D):
                                    sq[i] = Xs[i] * Xs[i]
                                T.reduce_sum(sq, sum_out, dim=0, clear=True)
                                rstd_k = T.rsqrt(sum_out[0] / D + eps)
                                for i in T.Parallel(D):
                                    Xs[i] = Xs[i] * rstd_k * KW[i].astype("float32")
                                for i in T.Parallel(D):
                                    pair = T.if_then_else(i < D // 2, i + D // 2, i - D // 2)
                                    xp = Xs[pair]
                                    cos = COS[pos, i].astype("float32")
                                    sin = SIN[pos, i].astype("float32")
                                    out = T.if_then_else(
                                        i < D // 2,
                                        Xs[i] * cos - xp * sin,
                                        Xs[i] * cos + xp * sin,
                                    )
                                    OK[b, kh, i] = out.astype("bfloat16")
                                    KC[b, kh, pos, i] = out.astype("bfloat16")
                                    VC[b, kh, pos, i] = V[b, kh, i]

            return main

        return jit_fn()

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_norm_w: torch.Tensor,
        k_norm_w: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, NQ, NKV, D = self.batch, self.num_heads, self.num_kv_heads, self.head_dim
        assert q.is_cuda and q.dtype == torch.bfloat16
        assert q.shape == (B, NQ, D) and q.is_contiguous()
        assert k.shape == (B, NKV, D) and k.is_contiguous()
        assert v.shape == (B, NKV, D) and v.is_contiguous()
        assert q_norm_w.shape == (D,) and q_norm_w.is_contiguous()
        assert k_norm_w.shape == (D,) and k_norm_w.is_contiguous()
        assert cos.shape == (self.max_seq_len, D) and cos.is_contiguous()
        assert sin.shape == (self.max_seq_len, D) and sin.is_contiguous()
        assert K_cache.shape == (B, NKV, self.max_seq_len, D) and K_cache.is_contiguous()
        assert V_cache.shape == K_cache.shape and V_cache.is_contiguous()

        if self._kernel is None:
            self._kernel = self._make_kernel()
        return self._kernel(q, k, v, q_norm_w, k_norm_w, cos, sin, K_cache, V_cache)
