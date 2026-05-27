import logging

import tilelang
import torch
from tilelang import language as T

logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("gemv", "tilelang")
class GemvTileLangKernel(BaseKernel):
    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K
        self.block_size = 1024  # TODO: tune for this op
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        N = self.N
        K = self.K

        @tilelang.jit(out_idx=[2])
        def kernel(BLOCK: int):
            @T.prim_func
            def main(
                A: T.Buffer((1, K), "float16"),
                B: T.Buffer((K, N), "float16"),
                C: T.Buffer((1, N), "float16"),
            ):
                # TODO: implement the tilelang kernel for gemv
                with T.Kernel(1, threads=BLOCK) as bx:
                    raise NotImplementedError("TODO: implement tilelang kernel for gemv")
            return main
        return kernel

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        # TODO: implement the call logic for gemv
        raise NotImplementedError("TODO: implement tilelang __call__ for gemv")
