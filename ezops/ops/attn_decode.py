import torch
import torch.nn.attention
from torch.nn.attention import SDPBackend

from .base_op import Op
from ..registry import get_kernel


class AttnDecodeOp(Op):
    _params_desc = {
        "batch": "Batch size",
        "num_heads": "Number of attention heads",
        "seq_len": "KV cache sequence length",
        "head_dim": "Dimension per head",
        "num_kv_heads": "Number of KV heads for GQA (default: num_heads)",
    }

    _atol = 1e-2
    _rtol = 1e-2

    def __init__(
        self,
        batch: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        backend: str = "ref",
        num_kv_heads: int | None = None,
    ):
        self.batch = batch
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )
        self._backend = backend
        if backend != "ref":
            kernel_cls = get_kernel("attn_decode", backend)
            if not getattr(kernel_cls, "supports_gqa", False):
                if self.num_kv_heads != num_heads:
                    raise ValueError(
                        f"backend {backend!r} does not support GQA "
                        f"(num_kv_heads={self.num_kv_heads} != num_heads={num_heads}); "
                        "expand K/V to num_heads before calling it"
                    )
                self._kernel = kernel_cls(batch, num_heads, seq_len, head_dim)
            else:
                self._kernel = kernel_cls(batch, num_heads, seq_len, head_dim, num_kv_heads=self.num_kv_heads)
        else:
            self._kernel = self._ref_forward

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self._kernel(Q, K, V)

    def gen_data(self):
        Q = torch.randn(self.batch, self.num_heads, 1, self.head_dim, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(self.batch, self.num_kv_heads, self.seq_len, self.head_dim, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(self.batch, self.num_kv_heads, self.seq_len, self.head_dim, device="cuda", dtype=torch.bfloat16)
        return Q, K, V

    def _ref_forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        with torch.nn.attention.sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(Q, K, V, enable_gqa=True)
        return out
