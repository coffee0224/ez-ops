from pathlib import Path

import torch

from tvm_ffi import cpp

from ..base_kernel import BaseKernel
from ...registry import register_kernel

_CU_SRC = Path(__file__).parent / "rmsnorm_cu.cu"


@register_kernel("rmsnorm", "cuda")
class RmsNormCudaKernel(BaseKernel):
    def __init__(self, batch_size: int, dim: int, eps: float = 1e-6):
        self.batch_size = batch_size
        self.dim = dim
        self.eps = eps
        self._mod = cpp.load_inline(
            name="rmsnorm_cuda",
            cuda_sources=_CU_SRC.read_text(),
            functions="rmsnorm_cu",
            extra_cuda_cflags=["-O3", "--generate-line-info"],
        )

    def __call__(self, X: torch.Tensor, Out: torch.Tensor) -> None:
        self._mod.rmsnorm_cu(X, Out, self.eps)
