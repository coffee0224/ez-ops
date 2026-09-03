import torch
import triton
import triton.language as tl

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("softmax", "triton")
class SoftmaxTritonKernel(BaseKernel):
    def __init__(self, batch_size: int, dim: int):
        self.batch_size = batch_size
        self.dim = dim
        self.block_size = 1024

    @staticmethod
    @triton.jit
    def _kernel(X_ptr, Out_ptr, dim, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        base = row * dim
        # pass 1: row max (masked lanes are -inf so they never win)
        m = tl.full((BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
        for start in range(0, dim, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X_ptr + base + offs, mask=offs < dim, other=float("-inf"))
            m = tl.maximum(m, x)
        m_row = tl.max(m, axis=0)
        # pass 2: sum of exp(x - max); masked lanes contribute exp(-inf) = 0
        s = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start in range(0, dim, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X_ptr + base + offs, mask=offs < dim, other=float("-inf"))
            s += tl.exp(x - m_row)
        inv = 1.0 / tl.sum(s, axis=0)
        # pass 3: normalize
        for start in range(0, dim, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X_ptr + base + offs, mask=offs < dim, other=0.0)
            tl.store(Out_ptr + base + offs, tl.exp(x - m_row) * inv, mask=offs < dim)

    def __call__(self, X: torch.Tensor, Out: torch.Tensor) -> None:
        assert X.is_cuda and Out.is_cuda
        assert X.shape == (self.batch_size, self.dim) and Out.shape == X.shape
        grid = lambda meta: (self.batch_size,)
        self._kernel[grid](X, Out, self.dim, BLOCK_SIZE=self.block_size)
