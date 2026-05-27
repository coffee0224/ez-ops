import torch
import triton
import triton.language as tl

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("gemv", "triton")
class GemvTritonKernel(BaseKernel):
    def __init__(self, M: int, N: int):
        self.M = M
        self.N = N
        self.block_size = 1024  # TODO: tune for this op

    @staticmethod
    @triton.jit
    def _kernel(
        A_ptr, x_ptr, y_ptr, M, N, stride_am,
        BLOCK_SIZE: tl.constexpr,
    ):
        # TODO: implement the triton kernel for gemv
        raise NotImplementedError("TODO: implement triton kernel for gemv")

    def __call__(self, A: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
        assert A.is_cuda and x.is_cuda and y.is_cuda
        assert A.shape == (self.M, self.N)
        assert x.shape == (self.N,)
        assert y.shape == (self.M,)
        stride_am, _ = A.stride()
        grid = lambda meta: (self.M,)
        self._kernel[grid](A, x, y, self.M, self.N, stride_am, BLOCK_SIZE=self.block_size)
