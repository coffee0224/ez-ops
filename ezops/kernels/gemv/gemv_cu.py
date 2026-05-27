from pathlib import Path

import torch
from tvm_ffi import cpp

from ..base_kernel import BaseKernel
from ...registry import register_kernel

_CU_SRC = Path(__file__).parent / "gemv_cu.cu"


@register_kernel("gemv", "cuda")
class GemvCudaKernel(BaseKernel):
    def __init__(self, M: int, N: int):
        self.M = M
        self.N = N
        self._mod = cpp.load_inline(
            name="gemv_cuda",
            cuda_sources=_CU_SRC.read_text(),
            functions="gemv_cu",
        )

    def __call__(self, A: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
        self._mod.gemv_cu(A, x, y)
