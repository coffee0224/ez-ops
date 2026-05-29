from pathlib import Path

import torch
from tvm_ffi import cpp

from ..base_kernel import BaseKernel
from ...registry import register_kernel

_CU_SRC = Path(__file__).parent / "gemv_cu.cu"


@register_kernel("gemv", "cuda")
class GemvCudaKernel(BaseKernel):
    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K
        self._mod = cpp.load_inline(
            name="gemv_cuda",
            cuda_sources=_CU_SRC.read_text(),
            functions="gemv_cu",
            extra_cuda_cflags=["-O3", "--generate-line-info"],
        )

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        self._mod.gemv_cu(A, B, C)
