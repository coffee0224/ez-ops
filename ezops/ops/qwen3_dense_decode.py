import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..registry import get_kernel
from .base_op import Op


class Qwen3DenseDecodeOp(Op):
    """One decode step of a Qwen3 dense decoder layer (batch x 1 token).

    Computes a single decoder-layer forward for the token being decoded,
    attending against a dense KV cache:

        residual = hidden
        x        = rmsnorm(hidden)                        # eps 1e-6
        q, k, v  = x @ Wqkv.T                             # fused rows [q | k | v]
        q        = rmsnorm(q, over head_dim)              # Qwen3 q_norm
        k        = rmsnorm(k, over head_dim)              # Qwen3 k_norm
        K_cache[:, :, -1] = k ; V_cache[:, :, -1] = v     # new KV into last slot
        attn     = GQA(q, K_cache, V_cache) / sqrt(head_dim)
        hidden   = residual + attn @ Wo.T
        residual = hidden
        x        = rmsnorm(hidden)
        hidden   = residual + (silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T

    RoPE is excluded, consistent with the framework's attn_decode op: cached
    K is treated as already rotated, and rotating the new token's q/k does
    not change the kernels' memory/compute profile.

    gen_data returns, in order (all bf16, cuda):
        hidden   [batch, hidden_size]
        K_cache  [batch, num_kv_heads, seq_len, head_dim]  # last slot: scratch-in, k-out
        V_cache  [batch, num_kv_heads, seq_len, head_dim]  # last slot: scratch-in, v-out
        Wqkv     [num_heads*D + 2*num_kv_heads*D, hidden_size]   # fused rows [q | k | v]
        Wo       [hidden_size, num_heads*D]
        Wgu      [2*intermediate_size, hidden_size]       # fused rows [gate | up]
        Wd       [hidden_size, intermediate_size]

    Weights arrive pre-fused (qkv / gate-up in one matrix) so backends stream
    each weight exactly once without re-stitching. The KV caches are mutated
    in place: slot seq_len-1 is reserved for the token being decoded. The
    write is idempotent for a fixed input, which keeps repeated benchmark
    iterations consistent.
    """

    _params_desc = {
        "batch": "Batch size (tokens per decode step)",
        "seq_len": "KV cache length; the last slot holds the token being decoded",
        "hidden_size": "Hidden dimension",
        "intermediate_size": "MLP intermediate dimension",
        "num_heads": "Number of query heads",
        "head_dim": "Dimension per head",
        "num_kv_heads": "Number of KV heads for GQA (default: num_heads)",
    }

    # Chained bf16 matmuls with per-stage rounding: backends that keep fp32
    # intermediates legitimately drift ~0.5% from a bf16-staged reference
    # (both sit ~0.3% from an fp32 oracle). Per-element atol doesn't fit, so
    # `check` uses relative Frobenius error, same as PdlGemmOp.
    _atol = 1e-2
    _rtol = 1e-2

    def check(self, actual, expected) -> bool:
        if actual.shape != expected.shape:
            return False
        diff = (actual - expected).norm().item()
        base = max(expected.norm().item(), 1.0)
        return (diff / base) < self._rtol

    rms_eps = 1e-6

    def __init__(
        self,
        batch: int,
        seq_len: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
        backend: str = "ref",
        num_kv_heads: int | None = None,
    ):
        if seq_len < 1:
            raise ValueError(f"seq_len ({seq_len}) must be >= 1")
        self.batch = batch
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")
        self.q_rows = num_heads * head_dim
        self.kv_rows = self.num_kv_heads * head_dim
        self._backend = backend
        if backend != "ref":
            kernel_cls = get_kernel("qwen3_dense_decode", backend)
            self._kernel = kernel_cls(
                batch,
                seq_len,
                hidden_size,
                intermediate_size,
                num_heads,
                head_dim,
                num_kv_heads=self.num_kv_heads,
            )
        else:
            self._kernel = self._ref_forward

    def forward(
        self,
        hidden: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        Wqkv: torch.Tensor,
        Wo: torch.Tensor,
        Wgu: torch.Tensor,
        Wd: torch.Tensor,
    ) -> torch.Tensor:
        return self._kernel(hidden, K_cache, V_cache, Wqkv, Wo, Wgu, Wd)

    def gen_data(self):
        dev = dict(device="cuda", dtype=torch.bfloat16)
        hidden = torch.randn(self.batch, self.hidden_size, **dev)
        K_cache = torch.randn(self.batch, self.num_kv_heads, self.seq_len, self.head_dim, **dev)
        V_cache = torch.randn(self.batch, self.num_kv_heads, self.seq_len, self.head_dim, **dev)
        # Xavier-style scaling keeps activations O(1) through the layer, so
        # bf16 comparisons stay meaningful. Scale by fan-in (cols).
        Wqkv = torch.randn(self.q_rows + 2 * self.kv_rows, self.hidden_size, **dev) * self.hidden_size**-0.5
        Wo = torch.randn(self.hidden_size, self.q_rows, **dev) * self.q_rows**-0.5
        Wgu = torch.randn(2 * self.intermediate_size, self.hidden_size, **dev) * self.hidden_size**-0.5
        Wd = torch.randn(self.hidden_size, self.intermediate_size, **dev) * self.intermediate_size**-0.5
        return hidden, K_cache, V_cache, Wqkv, Wo, Wgu, Wd

    def _rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        """Qwen3RMSNorm with unit weight: fp32 statistics, cast back to bf16."""
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.rms_eps)).to(x.dtype)

    def _ref_forward(
        self,
        hidden: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        Wqkv: torch.Tensor,
        Wo: torch.Tensor,
        Wgu: torch.Tensor,
        Wd: torch.Tensor,
    ) -> torch.Tensor:
        B = hidden.shape[0]
        qkv_rows = self.q_rows + 2 * self.kv_rows

        residual = hidden
        x = self._rmsnorm(hidden)
        qkv = x @ Wqkv.T
        q = self._rmsnorm(qkv[:, : self.q_rows].view(B, self.num_heads, self.head_dim))
        k = self._rmsnorm(qkv[:, self.q_rows : qkv_rows - self.kv_rows].view(B, self.num_kv_heads, self.head_dim))
        v = qkv[:, qkv_rows - self.kv_rows :].view(B, self.num_kv_heads, self.head_dim)

        K_cache[:, :, -1, :] = k
        V_cache[:, :, -1, :] = v

        # CUDNN rejects KV seqlen 1 (cache = only the token being decoded);
        # MATH covers that degenerate case, CUDNN takes everything else.
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]):
            attn = F.scaled_dot_product_attention(q.unsqueeze(2), K_cache, V_cache, enable_gqa=True).squeeze(2)
        hidden = residual + attn.reshape(B, self.q_rows) @ Wo.T

        residual = hidden
        x = self._rmsnorm(hidden)
        gu = x @ Wgu.T
        gate, up = gu[:, : self.intermediate_size], gu[:, self.intermediate_size :]
        hidden = residual + (F.silu(gate) * up) @ Wd.T
        return hidden
