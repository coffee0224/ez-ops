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
    }

    def __init__(self, batch: int, num_heads: int, seq_len: int, head_dim: int, backend: str = "ref"):
        self.batch = batch
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self._backend = backend
        if backend != "ref":
            kernel_cls = get_kernel("attn_decode", backend)
            self._kernel = kernel_cls(batch, num_heads, seq_len, head_dim)
        else:
            self._kernel = self._ref_forward

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self._kernel(Q, K, V)

    def gen_data(self):
        Q = torch.randn(self.batch, self.num_heads, 1, self.head_dim, device="cuda", dtype=torch.float32)
        K = torch.randn(self.batch, self.num_heads, self.seq_len, self.head_dim, device="cuda", dtype=torch.float32)
        V = torch.randn(self.batch, self.num_heads, self.seq_len, self.head_dim, device="cuda", dtype=torch.float32)
        return Q, K, V

    def _ref_forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]):
            out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        return out
