import logging

import tilelang
import torch
from tilelang import language as T

logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("gemv", "naive_gemv_tilelang")
class NaiveGemvTileLangKernel(BaseKernel):
    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        @tilelang.jit(out_idx=[2])
        def kernel(
            N: int,
            K: int,
            BLOCK_N: int,
            BLOCK_K: int,
            dtype: T.dtype = T.bfloat16,
            accum_dtype: T.dtype = T.float,
        ):
            @T.prim_func
            def main(
                A: T.Buffer((K,), dtype),
                B: T.Buffer((N, K), dtype),
                C: T.Buffer((N,), dtype),
            ):
                with T.Kernel(T.ceildiv(N, BLOCK_N)) as bn:
                    tn = T.get_thread_binding(0)  # tn = threadIdx.x
                    A_shared = T.alloc_shared((BLOCK_K,), dtype)
                    B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
                    C_reg = T.alloc_local((1,), accum_dtype)
                    T.clear(C_reg)
                    for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                        for tk in T.serial(BLOCK_K):
                            A_shared[tk] = A[bk * BLOCK_K + tk]
                            B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                        for tk in T.serial(BLOCK_K):
                            C_reg[0] += A_shared[tk].astype(accum_dtype) * B_shared[tn, tk].astype(accum_dtype)
                    C[bn * BLOCK_N + tn] = C_reg[0]

            return main

        return kernel

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        out = self._kernel(N=self.N, K=self.K, BLOCK_N=128, BLOCK_K=128)(A, B)
        C.copy_(out)
