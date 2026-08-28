import torch
import triton
import triton.language as tl

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("reduce", "triton")
class ReduceTritonKernel(BaseKernel):
    def __init__(self, n: int):
        self.n = n
        self.block_size = 1024

    @staticmethod
    @triton.jit
    def _kernel(A_ptr, Out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
        # cross-block accumulation via atomic; Out must be zeroed beforehand
        tl.atomic_add(Out_ptr, tl.sum(a, axis=0))

    def __call__(self, A: torch.Tensor, Out: torch.Tensor) -> None:
        assert A.is_cuda and Out.is_cuda
        assert A.shape == (self.n,) and Out.shape == (1,)
        Out.zero_()
        grid = lambda meta: (triton.cdiv(self.n, meta["BLOCK_SIZE"]),)
        self._kernel[grid](A, Out, self.n, BLOCK_SIZE=self.block_size)
