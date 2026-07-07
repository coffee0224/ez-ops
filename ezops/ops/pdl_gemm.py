import torch

from .base_op import Op
from ..registry import get_kernel


class PdlGemmOp(Op):
    """Two chained GEMMs: y = x @ W1, z = y @ W2.

    Used to demonstrate Programmatic Dependent Launch (PDL): overlapping two
    dependent GEMM kernels on the same stream to hide the second kernel's
    launch overhead and prolog behind the first kernel's epilogue.
    """

    _params_desc = {
        "M": "Batch / sequence dimension (rows of x, y, z)",
        "K": "Input feature dimension (cols of x, rows of W1)",
        "N": "Hidden dimension (cols of W1, rows of W2)",
        "P": "Output feature dimension (cols of W2)",
    }
    # Per-element atol from the base class doesn't fit a chained GEMM where
    # some backends quantize inputs to FP16/BF16 for tensor cores. Quantization
    # noise has a roughly constant floor relative to the largest output, so
    # elements near zero can have absolute error >> 0 while the overall
    # computation is correct. We override `check` to use relative Frobenius
    # norm error instead.
    _atol = 1e-2
    _rtol = 1e-2

    def check(self, actual, expected) -> bool:
        if actual.shape != expected.shape:
            return False
        diff = (actual - expected).norm().item()
        base = max(expected.norm().item(), 1.0)
        return (diff / base) < self._rtol

    def __init__(self, M: int, K: int, N: int, P: int, backend: str = "ref"):
        self.M = M
        self.K = K
        self.N = N
        self.P = P
        self._backend = backend
        if backend != "ref":
            kernel_cls = get_kernel("pdl_gemm", backend)
            self._kernel = kernel_cls(M=M, K=K, N=N, P=P)
        else:
            self._kernel = self._ref_forward

    def forward(
        self,
        x: torch.Tensor,
        W1: torch.Tensor,
        W2: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> None:
        self._kernel(x, W1, W2, y, z)

    def gen_data(self):
        x = torch.randn(self.M, self.K, device="cuda", dtype=torch.float32)
        W1 = torch.randn(self.K, self.N, device="cuda", dtype=torch.float32)
        W2 = torch.randn(self.N, self.P, device="cuda", dtype=torch.float32)
        y = torch.empty(self.M, self.N, device="cuda", dtype=torch.float32)
        z = torch.empty(self.M, self.P, device="cuda", dtype=torch.float32)
        return x, W1, W2, y, z

    def _ref_forward(
        self,
        x: torch.Tensor,
        W1: torch.Tensor,
        W2: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> None:
        torch.matmul(x, W1, out=y)
        torch.matmul(y, W2, out=z)
