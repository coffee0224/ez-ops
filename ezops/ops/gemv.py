import torch

from .base_op import Op
from ..registry import get_kernel


class GemvOp(Op):
    _params_desc = {"M": "Number of rows of matrix A", "N": "Number of columns of matrix A"}

    def __init__(self, M: int, N: int, backend: str = "triton"):
        self.M = M
        self.N = N
        self._backend = backend
        kernel_cls = get_kernel("gemv", backend)
        self._kernel = kernel_cls(M=M, N=N)

    def forward(self, A: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
        self._kernel(A, x, y)

    def gen_data(self):
        A = torch.randn(self.M, self.N, device="cuda", dtype=torch.float32)
        x = torch.randn(self.N, device="cuda", dtype=torch.float32)
        y = torch.empty(self.M, device="cuda", dtype=torch.float32)
        return A, x, y

    def _ref_forward(self, A: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
        y.copy_(A @ x)
