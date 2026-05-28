import logging
import itertools

import tilelang
import torch
from tilelang import language as T
from tilelang.autotuner import autotune
from tvm import DataType

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


@register_kernel("gemv", "naive_splitk_gemv_tilelang")
class NaiveSplitkGemvTileLangKernel(BaseKernel):
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
                with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, BLOCK_K)) as bn:
                    tn = T.get_thread_binding(0)
                    tk = T.get_thread_binding(1)
                    A_local = T.alloc_local((1,), dtype)
                    B_local = T.alloc_local((1,), dtype)
                    C_accum = T.alloc_local((1,), accum_dtype)
                    C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
                    if tk == 0:
                        C_shared[tn] = 0
                    T.clear(C_accum)
                    for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                        A_local[0] = A[bk * BLOCK_K + tk]
                        B_local[0] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                        C_accum[0] += A_local[0].astype(accum_dtype) * B_local[0].astype(accum_dtype)
                    T.atomic_add(C_shared[tn], C_accum[0])
                    C[bn * BLOCK_N + tn] = C_shared[tn]

            return main

        return kernel

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        out = self._kernel(N=self.N, K=self.K, BLOCK_N=32, BLOCK_K=32)(A, B)
        C.copy_(out)


@register_kernel("gemv", "splitk_gemv_tilelang")
class SplitkGemvTileLangKernel(BaseKernel):
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
            reduce_threads: int,
            dtype: T.dtype = T.bfloat16,
            accum_dtype: T.dtype = T.float,
        ):
            TILE_K = T.ceildiv(BLOCK_K, reduce_threads)

            @T.prim_func
            def main(
                A: T.Buffer((K,), dtype),
                B: T.Buffer((N, K), dtype),
                C: T.Buffer((N,), dtype),
            ):
                with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
                    tn = T.get_thread_binding(0)
                    tk = T.get_thread_binding(1)
                    A_local = T.alloc_local((TILE_K,), dtype)
                    B_local = T.alloc_local((TILE_K,), dtype)
                    C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
                    C_accum = T.alloc_local((1,), accum_dtype)
                    if tk == 0:
                        C_shared[tn] = 0
                    T.clear(C_accum)
                    for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                        for k in T.serial(TILE_K):
                            A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                            B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                        for k in T.serial(TILE_K):
                            C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
                    T.atomic_add(C_shared[tn], C_accum[0])
                    C[bn * BLOCK_N + tn] = C_shared[tn]

            return main

        return kernel

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        out = self._kernel(N=self.N, K=self.K, reduce_threads=32, BLOCK_N=32, BLOCK_K=32)(A, B)
        C.copy_(out)


@register_kernel("gemv", "splitk_gemv_vectorized_tilelang")
class SplitkGemvVectorizedTileLangKernel(BaseKernel):
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
            reduce_threads: int,
            dtype: T.dtype = T.bfloat16,
            accum_dtype: T.dtype = T.float,
        ):
            MAX_TRANSACTION_SIZE_IN_BITS = 128
            TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
            BLOCK_K = reduce_threads * TILE_K

            @T.prim_func
            def main(
                A: T.Buffer((K,), dtype),
                B: T.Buffer((N, K), dtype),
                C: T.Buffer((N,), dtype),
            ):
                with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
                    tn = T.get_thread_binding(0)
                    tk = T.get_thread_binding(1)
                    A_local = T.alloc_local((TILE_K,), dtype)
                    B_local = T.alloc_local((TILE_K,), dtype)
                    C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
                    C_accum = T.alloc_local((1,), accum_dtype)
                    if tk == 0:
                        C_shared[tn] = 0
                    T.clear(C_accum)
                    for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                        for k in T.vectorized(TILE_K):
                            A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                            B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                        for k in T.serial(TILE_K):
                            C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
                    T.atomic_add(C_shared[tn], C_accum[0])
                    C[bn * BLOCK_N + tn] = C_shared[tn]

            return main

        return kernel

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        out = self._kernel(N=self.N, K=self.K, reduce_threads=32, BLOCK_N=2)(A, B)
        C.copy_(out)


@register_kernel("gemv", "splitk_gemv_vectorized_tvm_tilelang")
class SplitkGemvVectorizedTvmTileLangKernel(BaseKernel):
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
            reduce_threads: int,
            dtype: T.dtype = T.bfloat16,
            accum_dtype: T.dtype = T.float,
        ):
            MAX_TRANSACTION_SIZE_IN_BITS = 128
            TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
            BLOCK_K = reduce_threads * TILE_K

            @T.prim_func
            def main(
                A: T.Buffer((K,), dtype),
                B: T.Buffer((N, K), dtype),
                C: T.Buffer((N,), dtype),
            ):
                with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
                    tn = T.get_thread_binding(0)
                    tk = T.get_thread_binding(1)
                    A_local = T.alloc_local((TILE_K,), dtype)
                    B_local = T.alloc_local((TILE_K,), dtype)
                    C_accum = T.alloc_local((1,), accum_dtype)

                    T.clear(C_accum)
                    for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                        for k in T.vectorized(TILE_K):
                            A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                            B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                        for k in T.serial(TILE_K):
                            C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
                    C_reduced = T.alloc_local((1,), accum_dtype)
                    with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.cast(0, accum_dtype)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                    ):
                        T.evaluate(
                            T.tvm_thread_allreduce(
                                T.uint32(1),
                                C_accum[0],
                                True,
                                C_reduced[0],
                                tk,
                                dtype="handle",
                            )
                        )

                    C[bn * BLOCK_N + tn] = C_reduced[0]

            return main

        return kernel

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        out = self._kernel(N=self.N, K=self.K, reduce_threads=32, BLOCK_N=2)(A, B)
        C.copy_(out)


@register_kernel("gemv", "autotune_tilelang")
class AutotuneGemvTileLangKernel(BaseKernel):
    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K
        self._kernel = self._make_kernel()
        self._best_kernel = None

    def _make_kernel(self):
        N = self.N
        K = self.K

        def get_configs():
            BLOCK_N = [2, 4, 8, 32, 64, 128]
            reduce_threads = [4, 8, 32]
            _configs = list(
                itertools.product(
                    BLOCK_N,
                    reduce_threads,
                )
            )
            configs = [
                {
                    "BLOCK_N": c[0],
                    "reduce_threads": c[1],
                }
                for c in _configs
            ]
            return configs

        @autotune(
            configs=get_configs(),
            warmup=3,
            rep=20,
        )
        @tilelang.jit(
            out_idx=[2],
            target="auto",
        )
        def kernel(
            BLOCK_N=None,
            reduce_threads=None,
        ):
            dtype = "bfloat16"
            accum_dtype = "float"
            if BLOCK_N is None or reduce_threads is None:
                BLOCK_N = 1
                reduce_threads = 1
            MAX_TRANSACTION_SIZE_IN_BITS = 128
            TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
            BLOCK_K = reduce_threads * TILE_K

            @T.prim_func
            def main(
                A: T.Buffer((K,), dtype),
                B: T.Buffer((N, K), dtype),
                C: T.Buffer((N,), dtype),
            ):
                with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
                    tn = T.get_thread_binding(0)
                    tk = T.get_thread_binding(1)
                    A_local = T.alloc_local((TILE_K,), dtype)
                    B_local = T.alloc_local((TILE_K,), dtype)
                    C_accum = T.alloc_local((1,), accum_dtype)

                    T.clear(C_accum)
                    for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                        for k in T.vectorized(TILE_K):
                            A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                            B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                        for k in T.serial(TILE_K):
                            C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
                    C_reduced = T.alloc_local((1,), accum_dtype)
                    with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.cast(0, accum_dtype)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                    ):
                        T.evaluate(
                            T.tvm_thread_allreduce(
                                T.uint32(1),
                                C_accum[0],
                                True,
                                C_reduced[0],
                                tk,
                                dtype="handle",
                            )
                        )

                    C[bn * BLOCK_N + tn] = C_reduced[0]

            return main

        return kernel

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        if self._best_kernel is None:
            self._best_kernel = self._kernel()
        out = self._best_kernel(A, B)
        C.copy_(out)


@register_kernel("gemv", "alloc_reducer_gemv_tilelang")
class AllocReducerGemvTileLangKernel(BaseKernel):
    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K
        self._kernel = self._make_kernel()
        self._best_kernel = None

    def _make_kernel(self):
        N = self.N
        K = self.K

        def get_configs():
            block_M = [32, 64, 128]
            block_N = [32, 64, 128]
            num_stages = [0, 1, 2]
            threads = [32, 64, 128]
            _configs = list(itertools.product(block_M, block_N, num_stages, threads))
            return [
                {
                    "block_M": c[0],
                    "block_N": c[1],
                    "num_stages": c[2],
                    "threads": c[3],
                }
                for c in _configs
            ]

        @autotune(
            configs=get_configs(),
            warmup=3,
            rep=20,
        )
        @tilelang.jit(
            pass_configs={
                tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
                tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            },
        )
        def kernel(A: T.Tensor, B: T.Tensor, block_M=None, block_N=None, num_stages=None, threads=None):
            dtype = "bfloat16"
            accum_dtype = "float"
            if block_M is None or block_N is None or num_stages is None or threads is None:
                block_M = 128
                block_N = 128
                num_stages = 2
                threads = 256

            A: T.Tensor((K,), dtype)
            B: T.Tensor((N, K), dtype)
            C = T.empty((N,), dtype)

            with T.Kernel(T.ceildiv(N, block_M), threads=threads) as i0_m:
                o_reducer = T.alloc_reducer(block_M, accum_dtype, replication="all")
                T.clear(o_reducer)
                for i0_n in T.Pipelined(T.ceildiv(K, block_N), num_stages=num_stages):
                    a_smem = T.alloc_shared((block_M, block_N), dtype)
                    T.copy(B[i0_m * block_M, i0_n * block_N], a_smem)
                    a_frag = T.alloc_fragment((block_M, block_N), dtype)
                    T.copy(a_smem, a_frag)
                    x_frag = T.alloc_fragment(block_N, dtype)
                    T.copy(A[i0_n * block_N], x_frag)
                    for i1_m, i1_n in T.Parallel(block_M, block_N):
                        o_reducer[i1_m] += a_frag[i1_m, i1_n].astype(accum_dtype) * x_frag[i1_n].astype(accum_dtype)
                T.finalize_reducer(o_reducer)
                T.copy(o_reducer, C[i0_m * block_M])

            return C

        return kernel

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        if self._best_kernel is None:
            # .compile() triggers autotune and returns the compiled JITKernel,
            # bypassing the autotuner's broken eager-mode execution path.
            self._best_kernel = self._kernel.compile(A, B)
        out = self._best_kernel(A, B)
        C.copy_(out)
