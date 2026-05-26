import logging

import tilelang
import torch
from tilelang import language as T

# tilelang's KernelCache warns "consider using @tilelang.jit" on cache hits,
# even when we already use @tilelang.jit. Suppress the misleading warning.
logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("vector_add", "tilelang")
class VectorAddTileLangKernel(BaseKernel):
    def __init__(self, n: int):
        self.n = n
        self.block_size = 1024
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        N = self.n

        @tilelang.jit(out_idx=[2])
        def kernel(BLOCK: int):
            @T.prim_func
            def main(
                A: T.Buffer((N,), "float32"),
                B: T.Buffer((N,), "float32"),
                C: T.Buffer((N,), "float32"),
            ):
                with T.Kernel(T.ceildiv(N, BLOCK), threads=BLOCK) as bx:
                    tx = T.get_thread_binding(0)
                    idx = bx * BLOCK + tx
                    if idx < N:
                        C[idx] = A[idx] + B[idx]

            return main

        return kernel

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        result = self._kernel(self.block_size)(A, B)
        C.copy_(result)
