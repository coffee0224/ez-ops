import torch

from .base_op import Op
from ..registry import get_kernel


class ReduceOp(Op):
    _params_desc = {"n": "Number of input elements"}

    def __init__(self, n: int, backend: str = "ref"):
        self.n = n
        self._backend = backend
        # fp32 reduction order differs across backends and the error grows
        # with N, so scale the absolute tolerance accordingly.
        self._atol = 1e-7 * n
        if backend != "ref":
            kernel_cls = get_kernel("reduce", backend)
            self._kernel = kernel_cls(n=n)
        else:
            self._kernel = self._ref_forward

    def forward(self, A: torch.Tensor, Out: torch.Tensor) -> None:
        self._kernel(A, Out)

    def gen_data(self):
        A = torch.randn(self.n, device="cuda", dtype=torch.float32)
        Out = torch.empty(1, device="cuda", dtype=torch.float32)
        return A, Out

    def _ref_forward(self, A: torch.Tensor, Out: torch.Tensor) -> None:
        Out.copy_(A.sum())
