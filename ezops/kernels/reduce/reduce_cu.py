from pathlib import Path

import torch

from tvm_ffi import cpp

from ..base_kernel import BaseKernel
from ...registry import register_kernel

_CU_SRC = Path(__file__).parent / "reduce_cu.cu"


@register_kernel("reduce", "cuda")
class ReduceCudaKernel(BaseKernel):
    def __init__(self, n: int):
        self.n = n
        self._mod = cpp.load_inline(
            name="reduce_cuda",
            cuda_sources=_CU_SRC.read_text(),
            functions="reduce_cu",
            extra_cuda_cflags=["-O3", "--generate-line-info"],
        )

    def __call__(self, A: torch.Tensor, Out: torch.Tensor) -> None:
        self._mod.reduce_cu(A, Out)
