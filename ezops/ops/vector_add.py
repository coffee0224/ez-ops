import torch

from .base_op import Op
from ..registry import get_kernel


class VectorAddOp(Op):
    _params_desc = {"n": "Number of elements"}

    def __init__(self, n: int, backend: str = "triton"):
        self.n = n
        self._backend = backend
        kernel_cls = get_kernel("vector_add", backend)
        self._kernel = kernel_cls(n=n)

    def forward(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        self._kernel(A, B, C)

    def gen_data(self):
        A = torch.randn(self.n, device="cuda", dtype=torch.float32)
        B = torch.randn(self.n, device="cuda", dtype=torch.float32)
        C = torch.empty(self.n, device="cuda", dtype=torch.float32)
        return A, B, C

    def _ref_forward(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        C.copy_(A + B)
