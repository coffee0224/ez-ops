import torch
import triton
import triton.language as tl

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("vector_add", "triton")
class VectorAddTritonKernel(BaseKernel):
    def __init__(self, n: int):
        self.n = n
        self.block_size = 1024

    @staticmethod
    @triton.jit
    def _kernel(A_ptr, B_ptr, C_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = tl.load(A_ptr + offsets, mask=mask)
        b = tl.load(B_ptr + offsets, mask=mask)
        tl.store(C_ptr + offsets, a + b, mask=mask)

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        assert A.shape == B.shape == C.shape == (self.n,)
        grid = lambda meta: (triton.cdiv(self.n, meta["BLOCK_SIZE"]),)
        self._kernel[grid](A, B, C, self.n, BLOCK_SIZE=self.block_size)
