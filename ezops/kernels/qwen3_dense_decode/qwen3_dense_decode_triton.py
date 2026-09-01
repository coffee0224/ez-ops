"""Triton backend for the qwen3_dense_decode op.

The decode layer is memory-bound at small batch (~100 MB of weights per
layer for qwen3-vl-2B vs a few KB of activations), so the backend fuses
every opportunity to avoid re-streaming tensors: norms are folded into the
GEMV that consumes them, and the SwiGLU activation is folded into the
down-projection GEMV. Five kernel launches:

  1. _qkv_rmsnorm_kernel            x=rmsnorm(hidden); qkv=x@Wqkv.T;
                                    q/k head-norm; -> QKV [B, q|k|v]
  2. _gqa_decode_kernel             online-softmax GQA decode; block 0 is the
                                    new token (k/v straight from QKV registers,
                                    written back to the cache's last slot),
                                    remaining blocks stream the cache
  3. _proj_residual_kernel          x2 = hidden + attn @ Wo.T
  4. _rmsnorm_gemm_kernel           xn=rmsnorm(x2); gu = xn @ Wgu.T
  5. _silu_mul_down_residual_kernel out = x2 + (silu(g)*u) @ Wd.T, with the
                                    activation recomputed per K-tile in fp32

All GEMVs accumulate in fp32 and use the sum-reduce form (tl.dot needs
M>=16, decode has M=1). Each CTA re-derives the input row's RMS statistic;
the row is only 2-4 KB so this is noise next to the weight tiles.
"""

import math

import torch
import triton
import triton.language as tl

from ...registry import register_kernel
from ..base_kernel import BaseKernel

RMS_EPS = 1e-6


@triton.jit
def _qkv_rmsnorm_kernel(
    X,
    W,
    QKV,
    stride_xb,
    stride_wb,
    stride_qkvb,
    H: tl.constexpr,
    D: tl.constexpr,
    NQ: tl.constexpr,
    NKV: tl.constexpr,
    eps,
    BLOCK_K: tl.constexpr,
):
    # One CTA per head: D weight rows -> D outputs. Global head id maps
    # directly onto the fused [q | k | v] row blocks of Wqkv.
    b = tl.program_id(0)
    head = tl.program_id(1)

    offs_h = tl.arange(0, H)
    x = tl.load(X + b * stride_xb + offs_h).to(tl.float32)
    rstd = tl.math.rsqrt(tl.sum(x * x) / H + eps)

    rows = head * D + tl.arange(0, D)
    acc = tl.zeros((D,), tl.float32)
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        xn = tl.load(X + b * stride_xb + offs_k).to(tl.float32) * rstd
        w = tl.load(W + rows[:, None] * stride_wb + offs_k[None, :]).to(tl.float32)
        acc += tl.sum(w * xn[None, :], axis=1)

    # Qwen3 applies RMSNorm over head_dim to q and k (not v).
    if head < NQ + NKV:
        acc = acc * tl.math.rsqrt(tl.sum(acc * acc) / D + eps)

    tl.store(QKV + b * stride_qkvb + rows, acc.to(tl.bfloat16))


@triton.jit(do_not_specialize=["S"])
def _gqa_decode_kernel(
    QKV,
    Kc,
    Vc,
    O,
    S,
    stride_qkvb,
    stride_kcb,
    stride_kch,
    stride_kcs,
    stride_vcb,
    stride_vch,
    stride_vcs,
    stride_ob,
    QOFF: tl.constexpr,
    KOFF: tl.constexpr,
    VOFF: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    sm_scale,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // GROUP
    d = tl.arange(0, D)

    q = tl.load(QKV + b * stride_qkvb + QOFF + h * D + d).to(tl.float32)

    # Block 0: the token being decoded. Its k/v come from QKV registers; the
    # first CTA of each KV group also persists them to the cache's last slot
    # (the decode step's KV write-back, ~KB, faithful to real inference).
    k_new = tl.load(QKV + b * stride_qkvb + KOFF + kvh * D + d).to(tl.float32)
    v_new = tl.load(QKV + b * stride_qkvb + VOFF + kvh * D + d).to(tl.float32)
    if h % GROUP == 0:
        last = (S - 1).to(tl.int64) * stride_kcs
        tl.store(Kc + b * stride_kcb + kvh * stride_kch + last + d, k_new.to(tl.bfloat16))
        last_v = (S - 1).to(tl.int64) * stride_vcs
        tl.store(Vc + b * stride_vcb + kvh * stride_vch + last_v + d, v_new.to(tl.bfloat16))

    m = tl.sum(q * k_new) * sm_scale
    l = 1.0
    acc = v_new

    # Remaining blocks: stream the cached slots [0, S-1).
    for s0 in range(0, S - 1, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask = offs_s < S - 1
        k = tl.load(
            Kc + b * stride_kcb + kvh * stride_kch + offs_s[:, None] * stride_kcs + d[None, :],
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)
        s = tl.sum(k * q[None, :], axis=1) * sm_scale
        s = tl.where(mask, s, -float("inf"))
        m_new = tl.maximum(m, tl.max(s, 0))
        p = tl.exp(s - m_new)
        v = tl.load(
            Vc + b * stride_vcb + kvh * stride_vch + offs_s[:, None] * stride_vcs + d[None, :],
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, 0)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        m = m_new

    tl.store(O + b * stride_ob + h * D + d, (acc / l).to(tl.bfloat16))


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
def _rmsnorm_gemm_kernel(
    X,
    W,
    O,
    H: tl.constexpr,
    stride_xb,
    stride_wb,
    stride_ob,
    eps,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # xn = rmsnorm(X[b, :]); O[b, rows] = xn @ W[rows, :].T   (gate/up proj)
    b = tl.program_id(0)
    offs_h = tl.arange(0, H)
    x = tl.load(X + b * stride_xb + offs_h).to(tl.float32)
    rstd = tl.math.rsqrt(tl.sum(x * x) / H + eps)

    rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        xn = tl.load(X + b * stride_xb + offs_k).to(tl.float32) * rstd
        w = tl.load(W + rows[:, None] * stride_wb + offs_k[None, :]).to(tl.float32)
        acc += tl.sum(w * xn[None, :], axis=1)
    tl.store(O + b * stride_ob + rows, acc.to(tl.bfloat16))


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


@register_kernel("qwen3_dense_decode", "triton")
class Qwen3DenseDecodeTritonKernel(BaseKernel):
    _block_k = 128  # K-tile of every GEMV loop
    _block_n = 64  # output rows per CTA for the 2048-row projections
    _block_n_wide = 128  # gate/up has 12288 rows: wider tiles, more CTAs
    _block_s = 128  # KV tile per attention iteration
    _num_warps = 4

    def __init__(
        self,
        batch: int,
        seq_len: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
    ):
        self.batch = batch
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = head_dim
        # tl.arange(0, H) for the full-row RMS pass needs powers of two, and
        # the GEMV loops need K divisible by the tile.
        for name, dim in (("hidden_size", hidden_size), ("head_dim", head_dim)):
            if dim & (dim - 1):
                raise ValueError(f"{name} ({dim}) must be a power of two")
        if hidden_size % self._block_k or intermediate_size % self._block_k:
            raise ValueError(
                f"hidden_size ({hidden_size}) and intermediate_size ({intermediate_size}) "
                f"must be divisible by BLOCK_K={self._block_k}"
            )
        self.q_rows = num_heads * head_dim
        self.kv_rows = self.num_kv_heads * head_dim
        self.qkv_rows = self.q_rows + 2 * self.kv_rows
        self.sm_scale = 1.0 / math.sqrt(head_dim)

    def __call__(
        self,
        hidden: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        Wqkv: torch.Tensor,
        Wo: torch.Tensor,
        Wgu: torch.Tensor,
        Wd: torch.Tensor,
    ) -> torch.Tensor:
        B, H, I = self.batch, self.hidden_size, self.intermediate_size
        assert hidden.is_cuda and hidden.dtype == torch.bfloat16
        assert hidden.shape == (B, H)
        assert K_cache.shape == (B, self.num_kv_heads, self.seq_len, self.head_dim)
        assert V_cache.shape == K_cache.shape
        assert Wqkv.shape == (self.qkv_rows, H) and Wqkv.stride(1) == 1
        assert Wo.shape == (H, self.q_rows) and Wo.stride(1) == 1
        assert Wgu.shape == (2 * I, H) and Wgu.stride(1) == 1
        assert Wd.shape == (H, I) and Wd.stride(1) == 1

        dev = hidden.device
        qkv = torch.empty((B, self.qkv_rows), device=dev, dtype=torch.bfloat16)
        attn = torch.empty((B, self.q_rows), device=dev, dtype=torch.bfloat16)
        x2 = torch.empty((B, H), device=dev, dtype=torch.bfloat16)
        gu = torch.empty((B, 2 * I), device=dev, dtype=torch.bfloat16)
        out = torch.empty((B, H), device=dev, dtype=torch.bfloat16)

        _qkv_rmsnorm_kernel[(B, self.num_heads + 2 * self.num_kv_heads)](
            hidden,
            Wqkv,
            qkv,
            hidden.stride(0),
            Wqkv.stride(0),
            qkv.stride(0),
            H=H,
            D=self.head_dim,
            NQ=self.num_heads,
            NKV=self.num_kv_heads,
            eps=RMS_EPS,
            BLOCK_K=self._block_k,
            num_warps=self._num_warps,
            num_stages=4,
        )
        _gqa_decode_kernel[(B, self.num_heads)](
            qkv,
            K_cache,
            V_cache,
            attn,
            self.seq_len,
            qkv.stride(0),
            K_cache.stride(0),
            K_cache.stride(1),
            K_cache.stride(2),
            V_cache.stride(0),
            V_cache.stride(1),
            V_cache.stride(2),
            attn.stride(0),
            QOFF=0,
            KOFF=self.q_rows,
            VOFF=self.q_rows + self.kv_rows,
            GROUP=self.num_heads // self.num_kv_heads,
            D=self.head_dim,
            sm_scale=self.sm_scale,
            BLOCK_S=self._block_s,
            num_warps=self._num_warps,
            num_stages=2,
        )
        _proj_residual_kernel[(B, triton.cdiv(H, self._block_n))](
            attn,
            Wo,
            hidden,
            x2,
            K=self.q_rows,
            stride_ab=attn.stride(0),
            stride_wb=Wo.stride(0),
            stride_rb=hidden.stride(0),
            stride_ob=x2.stride(0),
            BLOCK_N=self._block_n,
            BLOCK_K=self._block_k,
            num_warps=self._num_warps,
            num_stages=4,
        )
        _rmsnorm_gemm_kernel[(B, triton.cdiv(2 * I, self._block_n_wide))](
            x2,
            Wgu,
            gu,
            H=H,
            stride_xb=x2.stride(0),
            stride_wb=Wgu.stride(0),
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
