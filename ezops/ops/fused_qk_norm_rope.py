import torch

from ..registry import get_kernel
from .base_op import Op


class FusedQkNormRopeOp(Op):
    """Qwen3 per-head QK RMSNorm + RoPE + KV cache write-back for one decode step.

    Ported from mega-qwen's ``ldg_qk_norm_rope_cache`` decode phase; for each
    query head and each kv head of the token being decoded:

        q_head = rope(rmsnorm(q_head, q_norm_w),  cos[pos], sin[pos])
        k_head = rope(rmsnorm(k_head, k_norm_w),  cos[pos], sin[pos])
        K_cache[:, :, pos, :] = k_head ;  V_cache[:, :, pos, :] = v_head

    RoPE is the half-split (rotate_half) style over head_dim: for i < D/2 the
    output is x[i]*cos[i] - x[i+D/2]*sin[i], for i >= D/2 it is
    x[i]*cos[i] + x[i-D/2]*sin[i], indexed into full-width [max_seq_len,
    head_dim] cos/sin tables (HF layout, second half repeats the first).

    The reference keeps fp32 intermediates from norm through rotation with a
    single bf16 rounding at the outputs, matching the source kernel's fp32
    pipeline; statistics use eps 1e-6 over head_dim.

    gen_data returns, in order (all bf16, cuda):
        q         [batch, num_heads, head_dim]                      # pre-norm
        k         [batch, num_kv_heads, head_dim]                   # pre-norm
        v         [batch, num_kv_heads, head_dim]
        q_norm_w  [head_dim]
        k_norm_w  [head_dim]
        cos       [max_seq_len, head_dim]   # theta=1e6 (Qwen3), halves repeated
        sin       [max_seq_len, head_dim]
        K_cache   [batch, num_kv_heads, max_seq_len, head_dim]  # slot pos: k-out
        V_cache   [batch, num_kv_heads, max_seq_len, head_dim]  # slot pos: v-out

    The KV caches are mutated in place at slot ``position``: it holds random
    scratch on input and the normed/rotated k (raw v) on output. The write is
    idempotent for fixed inputs, which keeps repeated benchmark iterations
    consistent. Returns ``(q_out, k_out)``.
    """

    _params_desc = {
        "batch": "Batch size (tokens per decode step)",
        "num_heads": "Number of query heads",
        "head_dim": "Dimension per head (must be a power of two)",
        "max_seq_len": "KV cache capacity; also the cos/sin table length",
        "position": "Decode position: RoPE table row and cache slot to write",
        "num_kv_heads": "Number of KV heads for GQA (default: num_heads)",
    }

    # Norm and rotation run in fp32 with one bf16 rounding at the end, so ref
    # and backends differ only in reduction order (~1 bf16 ulp on O(1) values).
    _atol = 1e-2
    _rtol = 1e-2

    rms_eps = 1e-6
    rope_theta = 1e6  # Qwen3

    def __init__(
        self,
        batch: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        position: int,
        backend: str = "ref",
        num_kv_heads: int | None = None,
    ):
        if head_dim & (head_dim - 1):
            raise ValueError(f"head_dim ({head_dim}) must be a power of two")
        if not 0 <= position < max_seq_len:
            raise ValueError(f"position ({position}) must be in [0, {max_seq_len})")
        self.batch = batch
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.position = position
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")
        self._backend = backend
        if backend != "ref":
            kernel_cls = get_kernel("fused_qk_norm_rope", backend)
            self._kernel = kernel_cls(
                batch,
                num_heads,
                head_dim,
                max_seq_len,
                position,
                num_kv_heads=self.num_kv_heads,
            )
        else:
            self._kernel = self._ref_forward

    def forward(
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
        return self._kernel(q, k, v, q_norm_w, k_norm_w, cos, sin, K_cache, V_cache)

    def gen_data(self):
        dev = dict(device="cuda", dtype=torch.bfloat16)
        q = torch.randn(self.batch, self.num_heads, self.head_dim, **dev)
        k = torch.randn(self.batch, self.num_kv_heads, self.head_dim, **dev)
        v = torch.randn(self.batch, self.num_kv_heads, self.head_dim, **dev)
        q_norm_w = torch.randn(self.head_dim, **dev)
        k_norm_w = torch.randn(self.head_dim, **dev)
        # HF-style full-width tables: row `position` holds head_dim values whose
        # second half repeats the first, matching the rotate_half indexing.
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device="cuda") / self.head_dim)
        )
        angles = torch.arange(self.max_seq_len, dtype=torch.float32, device="cuda")[:, None] * inv_freq[None, :]
        cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(torch.bfloat16)
        sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(torch.bfloat16)
        K_cache = torch.randn(self.batch, self.num_kv_heads, self.max_seq_len, self.head_dim, **dev)
        V_cache = torch.randn(self.batch, self.num_kv_heads, self.max_seq_len, self.head_dim, **dev)
        return q, k, v, q_norm_w, k_norm_w, cos, sin, K_cache, V_cache

    def check(self, actual, expected) -> bool:
        for a, e in zip(actual, expected, strict=True):
            if a.shape != e.shape:
                return False
            if not torch.allclose(a, e, atol=self._atol, rtol=self._rtol):
                return False
        return True

    def _rmsnorm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Qwen3RMSNorm over the last dim with a per-dim weight; stays fp32."""
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return x.to(torch.float32) * torch.rsqrt(variance + self.rms_eps) * w.to(torch.float32)

    def _rope(self, x: torch.Tensor, cos_pos: torch.Tensor, sin_pos: torch.Tensor) -> torch.Tensor:
        """Half-split RoPE of an fp32 head against fp32 cos/sin rows."""
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos_pos + rotated * sin_pos).to(torch.bfloat16)

    def _ref_forward(
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
        cos_pos = cos[self.position].to(torch.float32)
        sin_pos = sin[self.position].to(torch.float32)

        q_out = self._rope(self._rmsnorm(q, q_norm_w), cos_pos, sin_pos)
        k_out = self._rope(self._rmsnorm(k, k_norm_w), cos_pos, sin_pos)

        K_cache[:, :, self.position, :] = k_out
        V_cache[:, :, self.position, :] = v
        return q_out, k_out
