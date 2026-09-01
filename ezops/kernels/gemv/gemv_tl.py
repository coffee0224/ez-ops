import logging
import itertools

import tilelang
import torch
from tilelang import language as T
from tilelang.autotuner import autotune
from tilelang.carver.arch.driver.cuda_driver import get_num_sms
from tilelang import PassConfigKey
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
        @tilelang.jit(out_idx=None)
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
                with T.Kernel(T.ceildiv(N, BLOCK_N), threads=BLOCK_N) as bn:
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
        self._kernel(N=self.N, K=self.K, BLOCK_N=128, BLOCK_K=128)(A, B, C)


@register_kernel("gemv", "naive_splitk_gemv_tilelang")
class NaiveSplitkGemvTileLangKernel(BaseKernel):
    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        @tilelang.jit(out_idx=None)
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
        self._kernel(N=self.N, K=self.K, BLOCK_N=32, BLOCK_K=32)(A, B, C)


@register_kernel("gemv", "splitk_gemv_tilelang")
class SplitkGemvTileLangKernel(BaseKernel):
    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        @tilelang.jit(out_idx=None)
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
        self._kernel(N=self.N, K=self.K, reduce_threads=32, BLOCK_N=32, BLOCK_K=32)(A, B, C)


@register_kernel("gemv", "splitk_gemv_vectorized_tilelang")
class SplitkGemvVectorizedTileLangKernel(BaseKernel):
    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        @tilelang.jit(out_idx=None)
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
        self._kernel(N=self.N, K=self.K, reduce_threads=32, BLOCK_N=2)(A, B, C)


@register_kernel("gemv", "splitk_gemv_vectorized_warp_reduce_tilelang")
class SplitkGemvVectorizedTvmTileLangKernel(BaseKernel):
    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        @tilelang.jit(out_idx=None)
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
                with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(reduce_threads, BLOCK_N)) as bn:
                    tk = T.get_thread_binding(0)
                    tn = T.get_thread_binding(1)
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

                    C[bn * BLOCK_N + tn] = T.warp_reduce_sum(C_accum[0])

            return main

        return kernel

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        self._kernel(N=self.N, K=self.K, reduce_threads=32, BLOCK_N=2)(A, B, C)


@register_kernel("gemv", "splitk_gemv_vectorized_warp_reduce_autotune_tilelang")
class SplitkGemvVectorizedWarpReduceAutotuneTileLangKernel(BaseKernel):
    """Autotuned warp-reduce gemv.

    reduce_threads is pinned to 32: tl::warp_reduce_sum is an unmasked
    full-warp butterfly reduce, and the reduce dim is threadIdx.x, so any
    value < 32 mixes adjacent rows' partial sums (BN >= 2) or relies on
    inactive lanes reading 0 (BN = 1, UB), and any value > 32 sums only
    part of the row. BLOCK_N is the only real tuning knob: it must keep
    grid = N / BLOCK_N large enough to fill all SMs (see
    scripts/sweep_gemv_warp_reduce.py).
    """

    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K
        self._kernel = self._make_kernel()
        self._best_kernel = None

    def _make_kernel(self):
        N = self.N
        K = self.K

        def get_configs():
            # BLOCK_N >= 32 (1024 threads/block) never wins and underfills
            # small grids; BLOCK_N must divide N (kernel has no bounds check).
            block_ns = [bn for bn in [1, 2, 4, 8, 16] if N % bn == 0]
            return [{"BLOCK_N": bn, "reduce_threads": 32} for bn in block_ns]

        @autotune(
            configs=get_configs(),
            warmup=3,
            rep=20,
        )
        @tilelang.jit(
            out_idx=None,
            target="auto",
        )
        def kernel(
            BLOCK_N: int = None,
            reduce_threads: int = None,
        ):
            dtype = "bfloat16"
            accum_dtype = "float"
            if BLOCK_N is None or reduce_threads is None:
                BLOCK_N = 1
                reduce_threads = 32
            MAX_TRANSACTION_SIZE_IN_BITS = 128
            TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
            BLOCK_K = reduce_threads * TILE_K

            @T.prim_func
            def main(
                A: T.Buffer((K,), dtype),
                B: T.Buffer((N, K), dtype),
                C: T.Buffer((N,), dtype),
            ):
                with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(reduce_threads, BLOCK_N)) as bn:
                    tk = T.get_thread_binding(0)
                    tn = T.get_thread_binding(1)
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

                    C[bn * BLOCK_N + tn] = T.warp_reduce_sum(C_accum[0])

            return main

        return kernel

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        if self._best_kernel is None:
            self._best_kernel = self._kernel()
        self._best_kernel(A, B, C)


@register_kernel("gemv", "ps_gemv_tilelang")
class PersistantGemvTilelangKernel(BaseKernel):
    """Persistent GEMV: grid=num_sms, shared A, coalesced B loads, warp-per-row."""

    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K
        self.num_sms = get_num_sms()
        self._kernel = self._make_kernel()
        self._best_kernel = None

    def _make_kernel(self):
        N = self.N
        K = self.K
        num_sms = self.num_sms

        def get_configs():
            return [
                {
                    "BLOCK_N": 8,
                    "BLOCK_K": 256,
                    "reduce_threads": 32,
                },
                {
                    "BLOCK_N": 16,
                    "BLOCK_K": 256,
                    "reduce_threads": 32,
                },
                {
                    "BLOCK_N": 16,
                    "BLOCK_K": 128,
                    "reduce_threads": 32,
                },
                {
                    "BLOCK_N": 8,
                    "BLOCK_K": 128,
                    "reduce_threads": 32,
                },
            ]

        @autotune(
            configs=get_configs(),
            warmup=3,
            rep=20,
        )
        @tilelang.jit(
            out_idx=None,
            target="auto",
        )
        def kernel(
            BLOCK_N: int = None,
            BLOCK_K: int = None,
            reduce_threads: int = None,
        ):
            dtype = "bfloat16"
            accum_dtype = "float"

            if BLOCK_N is None or BLOCK_K is None or reduce_threads is None:
                BLOCK_N = 8
                BLOCK_K = 256
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
                                    C_accum[0] += s_A[bk * BLOCK_K + lane_id * TILE_K + k].astype(
                                        accum_dtype
                                    ) * B_local[k].astype(accum_dtype)

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
        if self._best_kernel is None:
            # .compile() triggers autotune and returns the compiled JITKernel,
            # bypassing the autotuner's broken eager-mode execution path.
            self._best_kernel = self._kernel()
        self._best_kernel(A, B, C)
