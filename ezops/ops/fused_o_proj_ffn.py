import torch
import torch.nn.functional as F

from ..registry import get_kernel
from .base_op import Op


class FusedOProjFfnOp(Op):
    """Post-attention tail of a Qwen3 dense decoder layer (batch x 1 token).

    Ported from mega-qwen's ``ldg_o_proj_postnorm_mlp`` decode phase:

        x2   = residual + attn @ Wo.T                  # o_proj + residual
        xn   = rmsnorm(x2) * post_norm_w               # eps 1e-6
        out  = x2 + (silu(xn @ Wg.T) * (xn @ Wu.T)) @ Wd.T

    The o_proj input width (num_heads * head_dim) is independent of
    hidden_size (Qwen3-0.6B: 2048 attention lanes into a 1024 hidden).

    gen_data returns, in order (all bf16, cuda):
        attn         [batch, num_heads * head_dim]   # attention output
        residual     [batch, hidden_size]
        Wo           [hidden_size, num_heads * head_dim]
        post_norm_w  [hidden_size]
        Wg           [intermediate_size, hidden_size]
        Wu           [intermediate_size, hidden_size]
        Wd           [hidden_size, intermediate_size]

    Gate/up arrive as separate matrices (not pre-fused like
    Qwen3DenseDecodeOp's Wgu), matching the source kernel's weight layout;
    backends stream each weight exactly once. The op is stateless: the
    residual consumed and the residual produced by the source kernel's
    bookkeeping both collapse into ``out``.
    """

    _params_desc = {
        "batch": "Batch size (tokens per decode step)",
        "hidden_size": "Hidden dimension",
        "intermediate_size": "MLP intermediate dimension",
        "num_heads": "Number of query heads (o_proj input width = num_heads * head_dim)",
        "head_dim": "Dimension per head",
    }

    # Chained bf16 matmuls with per-stage rounding: backends that keep fp32
    # intermediates legitimately drift ~0.5% from a bf16-staged reference
    # (both sit ~0.3% from an fp32 oracle). Per-element atol doesn't fit, so
    # `check` uses relative Frobenius error, same as Qwen3DenseDecodeOp.
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
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
        backend: str = "ref",
    ):
        self.batch = batch
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_rows = num_heads * head_dim
        self._backend = backend
        if backend != "ref":
            kernel_cls = get_kernel("fused_o_proj_ffn", backend)
            self._kernel = kernel_cls(batch, hidden_size, intermediate_size, num_heads, head_dim)
        else:
            self._kernel = self._ref_forward

    def forward(
        self,
        attn: torch.Tensor,
        residual: torch.Tensor,
        Wo: torch.Tensor,
        post_norm_w: torch.Tensor,
        Wg: torch.Tensor,
        Wu: torch.Tensor,
        Wd: torch.Tensor,
    ) -> torch.Tensor:
        return self._kernel(attn, residual, Wo, post_norm_w, Wg, Wu, Wd)

    def gen_data(self):
        dev = dict(device="cuda", dtype=torch.bfloat16)
        # Xavier-style scaling keeps activations O(1) through the stage, so
        # bf16 comparisons stay meaningful. Scale by fan-in (cols).
        attn = torch.randn(self.batch, self.q_rows, **dev)
        residual = torch.randn(self.batch, self.hidden_size, **dev)
        Wo = torch.randn(self.hidden_size, self.q_rows, **dev) * self.q_rows**-0.5
        post_norm_w = torch.randn(self.hidden_size, **dev)
        Wg = torch.randn(self.intermediate_size, self.hidden_size, **dev) * self.hidden_size**-0.5
        Wu = torch.randn(self.intermediate_size, self.hidden_size, **dev) * self.hidden_size**-0.5
        Wd = torch.randn(self.hidden_size, self.intermediate_size, **dev) * self.intermediate_size**-0.5
        return attn, residual, Wo, post_norm_w, Wg, Wu, Wd

    def _rmsnorm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Qwen3RMSNorm with a per-dim weight: fp32 statistics, cast back to bf16."""
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.rms_eps) * w).to(x.dtype)

    def _ref_forward(
        self,
        attn: torch.Tensor,
        residual: torch.Tensor,
        Wo: torch.Tensor,
        post_norm_w: torch.Tensor,
        Wg: torch.Tensor,
        Wu: torch.Tensor,
        Wd: torch.Tensor,
    ) -> torch.Tensor:
        x2 = residual + attn @ Wo.T
        xn = self._rmsnorm(x2, post_norm_w)
        gate, up = xn @ Wg.T, xn @ Wu.T
        return x2 + (F.silu(gate) * up) @ Wd.T
