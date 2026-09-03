import torch

from .base_op import Op
from ..registry import get_kernel


class SoftmaxOp(Op):
    _params_desc = {
        "batch_size": "Number of rows",
        "dim": "Feature dimension per row (softmax over dim)",
    }
    _atol = 1e-6
    _rtol = 1e-5

    def __init__(self, batch_size: int, dim: int, backend: str = "ref"):
        self.batch_size = batch_size
        self.dim = dim
        self._backend = backend
        if backend != "ref":
            kernel_cls = get_kernel("softmax", backend)
            self._kernel = kernel_cls(batch_size=batch_size, dim=dim)
        else:
            self._kernel = self._ref_forward

    def forward(self, X: torch.Tensor, Out: torch.Tensor) -> None:
        self._kernel(X, Out)

    def gen_data(self):
        X = torch.randn(self.batch_size, self.dim, device="cuda", dtype=torch.float32)
        Out = torch.empty(self.batch_size, self.dim, device="cuda", dtype=torch.float32)
        return X, Out

    def _ref_forward(self, X: torch.Tensor, Out: torch.Tensor) -> None:
        Out.copy_(torch.softmax(X, dim=-1))
