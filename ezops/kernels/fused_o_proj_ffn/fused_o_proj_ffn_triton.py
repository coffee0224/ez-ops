"""Triton backend for the fused_o_proj_ffn op.

Port of mega-qwen's ``ldg_o_proj_postnorm_mlp`` decode phase. The stage is
memory-bound at decode batch (~25 MB of weights for qwen3-0.6B vs a few KB of
activations), so the backend fuses every opportunity to avoid re-streaming
tensors: the post-attention norm is folded into the gate/up GEMV that
consumes it, and the SwiGLU activation is folded into the down-projection
GEMV. Three kernel launches:

  1. _proj_residual_kernel          x2 = residual + attn @ Wo.T
  2. _rmsnorm_dual_gemm_kernel      xn = rmsnorm(x2) * post_norm_w;
                                    g/u = xn @ Wg.T / xn @ Wu.T -> GU [B, g|u]
  3. _silu_mul_down_residual_kernel out = x2 + (silu(g) * u) @ Wd.T, with the
                                    activation recomputed per K-tile in fp32

All GEMVs accumulate in fp32 and use the sum-reduce form (tl.dot needs
M>=16, decode has M=1). Each CTA re-derives the input row's RMS statistic;
the row is only 2-4 KB so this is noise next to the weight tiles.
"""

import torch
import triton
import triton.language as tl

from ...registry import register_kernel
from ..base_kernel import BaseKernel

RMS_EPS = 1e-6


@triton.jit
def _proj_residual_kernel(
    A,
    W,
    R,
    O,
    K: tl.constexpr,
    stride_ab,
    stride_wb,
    stride_rb,
    stride_ob,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # O[b, rows] = R[b, rows] + A[b, :] @ W[rows, :].T   (o_proj + residual)
    b = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A + b * stride_ab + offs_k).to(tl.float32)
        w = tl.load(W + rows[:, None] * stride_wb + offs_k[None, :]).to(tl.float32)
        acc += tl.sum(w * a[None, :], axis=1)
    r = tl.load(R + b * stride_rb + rows).to(tl.float32)
    tl.store(O + b * stride_ob + rows, (acc + r).to(tl.bfloat16))


@triton.jit
def _rmsnorm_dual_gemm_kernel(
    X,
    NW,
    WG,
    WU,
    O,
    H: tl.constexpr,
    I: tl.constexpr,
    stride_xb,
    stride_wb,
    stride_ob,
    eps,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # xn = rmsnorm(X[b, :]) * NW; O[b, rows] = xn @ WG[rows, :].T and
    # O[b, I + rows] = xn @ WU[rows, :].T — gate and up rows of the same
    # output share the normed input tile, like the source kernel's fused
    # gate/up matvec loop.
    b = tl.program_id(0)
    offs_h = tl.arange(0, H)
    x = tl.load(X + b * stride_xb + offs_h).to(tl.float32)
    rstd = tl.math.rsqrt(tl.sum(x * x) / H + eps)

    rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc_g = tl.zeros((BLOCK_N,), tl.float32)
    acc_u = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        xn = tl.load(X + b * stride_xb + offs_k).to(tl.float32) * rstd * tl.load(NW + offs_k).to(tl.float32)
        wg = tl.load(WG + rows[:, None] * stride_wb + offs_k[None, :]).to(tl.float32)
        wu = tl.load(WU + rows[:, None] * stride_wb + offs_k[None, :]).to(tl.float32)
        acc_g += tl.sum(wg * xn[None, :], axis=1)
        acc_u += tl.sum(wu * xn[None, :], axis=1)

    tl.store(O + b * stride_ob + rows, acc_g.to(tl.bfloat16))
    tl.store(O + b * stride_ob + I + rows, acc_u.to(tl.bfloat16))


@triton.jit
def _silu_mul_down_residual_kernel(
    GU,
    W,
    R,
    O,
    I: tl.constexpr,
    stride_gub,
    stride_wb,
    stride_rb,
    stride_ob,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # O[b, rows] = R[b, rows] + (silu(g) * u) @ W[rows, :].T where g, u are the
    # two halves of GU[b, :]. The activation is recomputed per K-tile in fp32.
    b = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for i0 in range(0, I, BLOCK_K):
        offs_i = i0 + tl.arange(0, BLOCK_K)
        g = tl.load(GU + b * stride_gub + offs_i).to(tl.float32)
        u = tl.load(GU + b * stride_gub + I + offs_i).to(tl.float32)
        act = g * tl.sigmoid(g) * u
        w = tl.load(W + rows[:, None] * stride_wb + offs_i[None, :]).to(tl.float32)
        acc += tl.sum(w * act[None, :], axis=1)
    r = tl.load(R + b * stride_rb + rows).to(tl.float32)
    tl.store(O + b * stride_ob + rows, (acc + r).to(tl.bfloat16))


@register_kernel("fused_o_proj_ffn", "triton")
class FusedOProjFfnTritonKernel(BaseKernel):
    _block_k = 128  # K-tile of every GEMV loop
    _block_n = 64  # output rows per CTA for the hidden-sized projections
    _block_n_wide = 128  # gate/up has 2*intermediate rows: wider tiles, more CTAs
    _num_warps = 4

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
        # tl.arange(0, H) for the full-row RMS pass needs powers of two, and
        # the GEMV loops need every K dimension divisible by the tile.
        if hidden_size & (hidden_size - 1):
            raise ValueError(f"hidden_size ({hidden_size}) must be a power of two")
        if hidden_size % self._block_k or intermediate_size % self._block_k or self.q_rows % self._block_k:
            raise ValueError(
                f"hidden_size ({hidden_size}), intermediate_size ({intermediate_size}) and "
                f"num_heads * head_dim ({self.q_rows}) must be divisible by BLOCK_K={self._block_k}"
            )

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
        assert attn.shape == (B, self.q_rows) and attn.stride(1) == 1
        assert residual.shape == (B, H) and residual.stride(1) == 1
        assert Wo.shape == (H, self.q_rows) and Wo.stride(1) == 1
        assert post_norm_w.shape == (H,) and post_norm_w.stride(0) == 1
        assert Wg.shape == (I, H) and Wg.stride(1) == 1
        assert Wu.shape == (I, H) and Wu.stride(1) == 1
        assert Wd.shape == (H, I) and Wd.stride(1) == 1

        dev = attn.device
        x2 = torch.empty((B, H), device=dev, dtype=torch.bfloat16)
        gu = torch.empty((B, 2 * I), device=dev, dtype=torch.bfloat16)
        out = torch.empty((B, H), device=dev, dtype=torch.bfloat16)

        _proj_residual_kernel[(B, triton.cdiv(H, self._block_n))](
            attn,
            Wo,
            residual,
            x2,
            K=self.q_rows,
            stride_ab=attn.stride(0),
            stride_wb=Wo.stride(0),
            stride_rb=residual.stride(0),
            stride_ob=x2.stride(0),
            BLOCK_N=self._block_n,
            BLOCK_K=self._block_k,
            num_warps=self._num_warps,
            num_stages=4,
        )
        _rmsnorm_dual_gemm_kernel[(B, triton.cdiv(I, self._block_n_wide))](
            x2,
            post_norm_w,
            Wg,
            Wu,
            gu,
            H=H,
            I=I,
            stride_xb=x2.stride(0),
            stride_wb=Wg.stride(0),
            stride_ob=gu.stride(0),
            eps=RMS_EPS,
            BLOCK_N=self._block_n_wide,
            BLOCK_K=self._block_k,
            num_warps=self._num_warps,
            num_stages=4,
        )
        _silu_mul_down_residual_kernel[(B, triton.cdiv(H, self._block_n))](
            gu,
            Wd,
            x2,
            out,
            I=I,
            stride_gub=gu.stride(0),
            stride_wb=Wd.stride(0),
            stride_rb=x2.stride(0),
            stride_ob=out.stride(0),
            BLOCK_N=self._block_n,
            BLOCK_K=self._block_k,
            num_warps=self._num_warps,
            num_stages=4,
        )
        return out
