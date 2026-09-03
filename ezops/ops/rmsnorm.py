import torch

from .base_op import Op
from ..registry import get_kernel


class RmsNormOp(Op):
    _params_desc = {
        "batch_size": "Number of rows",
        "dim": "Feature dimension per row (normalized over dim)",
    }
    _atol = 1e-5
    _rtol = 1e-5

    def __init__(self, batch_size: int, dim: int, backend: str = "ref", eps: float = 1e-6):
        self.batch_size = batch_size
        self.dim = dim
        self.eps = eps
        self._backend = backend
        if backend != "ref":
            kernel_cls = get_kernel("rmsnorm", backend)
            self._kernel = kernel_cls(batch_size=batch_size, dim=dim, eps=eps)
        else:
            self._kernel = self._ref_forward

    def forward(self, X: torch.Tensor, Out: torch.Tensor) -> None:
        self._kernel(X, Out)

    def gen_data(self):
        X = torch.randn(self.batch_size, self.dim, device="cuda", dtype=torch.float32)
        Out = torch.empty(self.batch_size, self.dim, device="cuda", dtype=torch.float32)
        return X, Out

    def _ref_forward(self, X: torch.Tensor, Out: torch.Tensor) -> None:
        ms = X.pow(2).mean(dim=-1, keepdim=True)
        Out.copy_(X * torch.rsqrt(ms + self.eps))
