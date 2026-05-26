from pathlib import Path

import torch
from tvm_ffi import cpp

from ..base_kernel import BaseKernel
from ...registry import register_kernel

_CU_SRC = Path(__file__).parent / "vector_add_cu.cu"


@register_kernel("vector_add", "cuda")
class VectorAddCudaKernel(BaseKernel):
    def __init__(self, n: int):
        self.n = n
        self._mod = cpp.load_inline(
            name="vector_add_cuda",
            cuda_sources=_CU_SRC.read_text(),
            functions="vector_add_cu",
        )

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        self._mod.vector_add_cu(A, B, C)
