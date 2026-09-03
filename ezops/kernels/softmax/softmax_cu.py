from pathlib import Path

import torch

from tvm_ffi import cpp

from ..base_kernel import BaseKernel
from ...registry import register_kernel

_CU_SRC = Path(__file__).parent / "softmax_cu.cu"


@register_kernel("softmax", "cuda")
class SoftmaxCudaKernel(BaseKernel):
    def __init__(self, batch_size: int, dim: int):
        self.batch_size = batch_size
        self.dim = dim
        self._mod = cpp.load_inline(
            name="softmax_cuda",
            cuda_sources=_CU_SRC.read_text(),
            functions="softmax_cu",
            extra_cuda_cflags=["-O3", "--generate-line-info"],
        )

    def __call__(self, X: torch.Tensor, Out: torch.Tensor) -> None:
        self._mod.softmax_cu(X, Out)
