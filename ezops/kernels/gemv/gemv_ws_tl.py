import logging

import tilelang
import torch
from tilelang import language as T
from tilelang.carver.arch.driver.cuda_driver import get_num_sms
from tilelang import PassConfigKey

logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("gemv", "ws_tilelang")
class GemvWsTilelangKernel(BaseKernel):
    """Persistent GEMV: grid=num_sms, shared A, coalesced B loads, warp-per-row."""

    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K
        self.num_sms = get_num_sms()
        self._kernel = self._make_kernel()
        self._compiled = None

    def _make_kernel(self):
        N = self.N
        K = self.K
        num_sms = self.num_sms
        BLOCK_N = 8
        BLOCK_K = 256

        @tilelang.jit(
            out_idx=[2],
            pass_configs={
                PassConfigKey.TL_DISABLE_TMA_LOWER: True,
                PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            },
        )
        def kernel(
            dtype: str = "bfloat16",
            accum_dtype: str = "float",
        ):
            reduce_threads = 32
            TILE_K = BLOCK_K // reduce_threads
            total_row_groups = T.ceildiv(N, BLOCK_N)
            num_iters = T.ceildiv(total_row_groups, num_sms)

            @T.prim_func
            def main(
                A: T.Buffer((K,), dtype),
                B: T.Buffer((N, K), dtype),
                C: T.Buffer((N,), dtype),
            ):
                with T.Kernel(num_sms, threads=(reduce_threads, BLOCK_N)) as block_id:
                    lane_id = T.get_thread_binding(0)
                    row_id = T.get_thread_binding(1)

                    s_A = T.alloc_shared((K,), dtype)
                    T.copy(A, s_A)

                    B_local = T.alloc_local((TILE_K,), dtype)
                    C_accum = T.alloc_local((1,), accum_dtype)
                    C_reduced = T.alloc_local((1,), accum_dtype)

                    for it in T.serial(num_iters):
                        rg = it * num_sms + block_id
                        if rg < total_row_groups:
                            my_row = rg * BLOCK_N + row_id
                            T.clear(C_accum)

                            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                                for k in T.vectorized(TILE_K):
                                    B_local[k] = B[my_row, bk * BLOCK_K + lane_id * TILE_K + k]
                                for k in T.serial(TILE_K):
                                    C_accum[0] += (
                                        s_A[bk * BLOCK_K + lane_id * TILE_K + k].astype(accum_dtype)
                                        * B_local[k].astype(accum_dtype)
                                    )

                            with T.attr(
                                T.comm_reducer(
                                    lambda x, y: x + y,
                                    [T.cast(0, accum_dtype)],
                                ),
                                "reduce_scope",
                                T.reinterpret(T.uint64(0), dtype="handle"),
                            ):
                                T.evaluate(
                                    T.tvm_thread_allreduce(
                                        T.uint32(1),
                                        C_accum[0],
                                        True,
                                        C_reduced[0],
                                        lane_id,
                                        dtype="handle",
                                    )
                                )
                            if lane_id == 0 and my_row < N:
                                C[my_row] = C_reduced[0]

            return main

        return kernel

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        if self._compiled is None:
            self._compiled = self._kernel()
        out = self._compiled(A, B)
        C.copy_(out)
