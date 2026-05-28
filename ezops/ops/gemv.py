import torch

from .base_op import Op
from ..registry import get_kernel


class GemvOp(Op):
    _params_desc = {
        "N": "Number of columns of C",
        "K": "Reduction dimension (A cols, B rows)",
    }
    _atol = 1e-2
    _rtol = 1e-2

    def __init__(self, N: int, K: int, backend: str = "ref"):
        self.N = N
        self.K = K
        self._backend = backend
        if backend != "ref":
            kernel_cls = get_kernel("gemv", backend)
            self._kernel = kernel_cls(N=N, K=K)
        else:
            self._kernel = self._ref_forward

    def forward(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        self._kernel(A, B, C)

    def gen_data(self):
        A = torch.randn(self.K, device="cuda", dtype=torch.bfloat16)
        if self._backend == "ref":
            B = torch.randn(self.K, self.N, device="cuda", dtype=torch.bfloat16)
        else:
            B = torch.randn(self.N, self.K, device="cuda", dtype=torch.bfloat16)
        C = torch.empty(self.N, device="cuda", dtype=torch.bfloat16)
        return A, B, C

    def _ref_forward(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        C.copy_(A @ B)
