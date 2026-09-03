import torch
import triton
import triton.language as tl

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("rmsnorm", "triton")
class RmsNormTritonKernel(BaseKernel):
    def __init__(self, batch_size: int, dim: int, eps: float = 1e-6):
        self.batch_size = batch_size
        self.dim = dim
        self.eps = eps
        self.block_size = 1024

    @staticmethod
    @triton.jit
    def _kernel(X_ptr, Out_ptr, dim, eps, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        base = row * dim
        # pass 1: sum of squares over the row, chunked to keep BLOCK_SIZE fixed
        sumsq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start in range(0, dim, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X_ptr + base + offs, mask=offs < dim, other=0.0)
            sumsq += x * x
        scale = tl.rsqrt(tl.sum(sumsq, axis=0) / dim + eps)
        # pass 2: normalize
        for start in range(0, dim, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X_ptr + base + offs, mask=offs < dim, other=0.0)
            tl.store(Out_ptr + base + offs, x * scale, mask=offs < dim)

    def __call__(self, X: torch.Tensor, Out: torch.Tensor) -> None:
        assert X.is_cuda and Out.is_cuda
        assert X.shape == (self.batch_size, self.dim) and Out.shape == X.shape
        grid = lambda meta: (self.batch_size,)
        self._kernel[grid](X, Out, self.dim, self.eps, BLOCK_SIZE=self.block_size)
