"""Triton backend for the fused_qk_norm_rope op.

Port of mega-qwen's ``ldg_qk_norm_rope_cache`` decode phase. The stage is
purely elementwise over heads (no weights larger than head_dim), so a single
launch handles everything: one CTA per (batch, head), heads [0, NQ) norm and
rotate a query head, heads [NQ, NQ+NKV) norm and rotate a kv head and also
persist k/v into the cache slot being decoded.

Norm statistics and rotation stay in fp32 with one bf16 rounding at each
store, matching the source kernel's fp32 pipeline.
"""

import torch
import triton
import triton.language as tl

from ...registry import register_kernel
from ..base_kernel import BaseKernel

RMS_EPS = 1e-6


@triton.jit
def _norm_rope_head(
    X,
    W,
    COS,
    SIN,
    position,
    eps,
    D: tl.constexpr,
):
    # RMSNorm over one head (per-dim weight) + half-split RoPE at `position`.
    # Returns the rotated head in fp32; each caller rounds once at its stores.
    offs = tl.arange(0, D)
    x = tl.load(X + offs).to(tl.float32)
    w = tl.load(W + offs).to(tl.float32)
    scale = tl.math.rsqrt(tl.sum(x * x) / D + eps)

    xn = x * scale * w

    # The rotation needs the partner lane's normed value (i +/- D/2). Heads
    # are L1-resident at D elements, so reload it permuted instead of
    # keeping both halves live through the reduction.
    pair = tl.where(offs < D // 2, offs + D // 2, offs - D // 2)
    xp = tl.load(X + pair).to(tl.float32) * scale * tl.load(W + pair).to(tl.float32)

    cos = tl.load(COS + position * D + offs).to(tl.float32)
    sin = tl.load(SIN + position * D + offs).to(tl.float32)
    return tl.where(offs < D // 2, xn * cos - xp * sin, xn * cos + xp * sin)


@triton.jit
def _qk_norm_rope_cache_kernel(
    Q,
    K,
    V,
    QW,
    KW,
    COS,
    SIN,
    KC,
    VC,
    OQ,
    OK,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_vb,
    stride_vh,
    stride_kcb,
    stride_kch,
    stride_kcs,
    stride_vcb,
    stride_vch,
    stride_vcs,
    stride_oqb,
    stride_oqh,
    stride_okb,
    stride_okh,
    position,
    eps,
    NQ: tl.constexpr,
    D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.arange(0, D)

    if h < NQ:
        out = _norm_rope_head(Q + b * stride_qb + h * stride_qh, QW, COS, SIN, position, eps, D)
        tl.store(OQ + b * stride_oqb + h * stride_oqh + offs, out.to(tl.bfloat16))
    else:
        # kv heads: the rotated k and the raw v also land in the cache slot
        # being decoded (the decode step's KV write-back).
        kh = h - NQ
        out = _norm_rope_head(K + b * stride_kb + kh * stride_kh, KW, COS, SIN, position, eps, D)
        tl.store(OK + b * stride_okb + kh * stride_okh + offs, out.to(tl.bfloat16))
        tl.store(KC + b * stride_kcb + kh * stride_kch + position * stride_kcs + offs, out.to(tl.bfloat16))
        v = tl.load(V + b * stride_vb + kh * stride_vh + offs)
        tl.store(VC + b * stride_vcb + kh * stride_vch + position * stride_vcs + offs, v)


@register_kernel("fused_qk_norm_rope", "triton")
class FusedQkNormRopeTritonKernel(BaseKernel):
    _num_warps = 1  # one head = D elements; a single warp vectorizes the loads

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
        assert q.shape == (B, NQ, D) and q.stride(2) == 1
        assert k.shape == (B, NKV, D) and k.stride(2) == 1
        assert v.shape == (B, NKV, D) and v.stride(2) == 1
        assert q_norm_w.shape == (D,) and q_norm_w.stride(0) == 1
        assert k_norm_w.shape == (D,) and k_norm_w.stride(0) == 1
        assert cos.shape == (self.max_seq_len, D) and cos.stride(1) == 1
        assert sin.shape == (self.max_seq_len, D) and sin.stride(1) == 1
        assert K_cache.shape == (B, NKV, self.max_seq_len, D)
        assert V_cache.shape == K_cache.shape

        oq = torch.empty_like(q)
        ok = torch.empty_like(k)

        _qk_norm_rope_cache_kernel[(B, NQ + NKV)](
            q,
            k,
            v,
            q_norm_w,
            k_norm_w,
            cos,
            sin,
            K_cache,
            V_cache,
            oq,
            ok,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            K_cache.stride(0),
            K_cache.stride(1),
            K_cache.stride(2),
            V_cache.stride(0),
            V_cache.stride(1),
            V_cache.stride(2),
            oq.stride(0),
            oq.stride(1),
            ok.stride(0),
            ok.stride(1),
            self.position,
            RMS_EPS,
            NQ=NQ,
            D=D,
            num_warps=self._num_warps,
        )
        return oq, ok
