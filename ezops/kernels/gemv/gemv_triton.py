import torch
import triton
import triton.language as tl

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("gemv", "triton")
class GemvTritonKernel(BaseKernel):
    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K
        self.block_size = 1024  # TODO: tune for this op

    @staticmethod
    @triton.jit
    def _kernel(
        A_ptr, B_ptr, C_ptr, N, K, stride_bk,
        BLOCK_SIZE: tl.constexpr,
    ):
        # TODO: implement the triton kernel for gemv
        raise NotImplementedError("TODO: implement triton kernel for gemv")

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        assert A.shape == (self.K,)
        assert B.shape == (self.N, self.K)
        assert C.shape == (self.N,)
        _, stride_bk = B.stride()
        grid = lambda meta: (1,)
        self._kernel[grid](A, B, C, self.N, self.K, stride_bk, BLOCK_SIZE=self.block_size)
