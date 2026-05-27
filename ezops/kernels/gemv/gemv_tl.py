import logging

import tilelang
import torch
from tilelang import language as T

logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("gemv", "tilelang")
class GemvTileLangKernel(BaseKernel):
    def __init__(self, M: int, N: int):
        self.M = M
        self.N = N
        self.block_size = 1024  # TODO: tune for this op
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        M = self.M
        N = self.N

        @tilelang.jit(out_idx=[2])
        def kernel(BLOCK: int):
            @T.prim_func
            def main(
                A: T.Buffer((M, N), "float32"),
                x: T.Buffer((N,), "float32"),
                y: T.Buffer((M,), "float32"),
            ):
                # TODO: implement the tilelang kernel for gemv
                with T.Kernel(M, threads=BLOCK) as bx:
                    raise NotImplementedError("TODO: implement tilelang kernel for gemv")
            return main
        return kernel

    def __call__(self, A: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
        assert A.is_cuda and x.is_cuda and y.is_cuda
        # TODO: implement the call logic for gemv
        raise NotImplementedError("TODO: implement tilelang __call__ for gemv")
