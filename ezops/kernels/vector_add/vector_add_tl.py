import torch

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("vector_add", "tilelang")
class VectorAddTileLangKernel(BaseKernel):
    def __init__(self, n: int):
        self.n = n
        self.block_size = 1024
        self._program = self._build()

    def _build(self):
        import tilelang as tl
        from tilelang import language as T

        N = self.n
        BLOCK = self.block_size

        @T.prim_func
        def vector_add(A: T.Buffer((N,), "float32"), B: T.Buffer((N,), "float32"), C: T.Buffer((N,), "float32")):
            with T.Kernel(T.ceildiv(N, BLOCK), threads=BLOCK) as bx:
                tx = T.get_thread_binding(0)
                idx = bx * BLOCK + tx
                if idx < N:
                    C[idx] = A[idx] + B[idx]

        return tl.compile(vector_add, out_idx=[2])

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        result = self._program(A, B)
        C.copy_(result)
