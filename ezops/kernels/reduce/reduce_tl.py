import logging

import tilelang
import torch
from tilelang import language as T

# tilelang's KernelCache warns "consider using @tilelang.jit" on cache hits,
# even when we already use @tilelang.jit. Suppress the misleading warning.
logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("reduce", "tilelang")
class ReduceTileLangKernel(BaseKernel):
    def __init__(self, n: int):
        self.n = n
        self.block_size = 256
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        N = self.n

        @tilelang.jit(out_idx=None)
        def kernel(BLOCK: int):
            @T.prim_func
            def main(
                A: T.Buffer((N,), "float32"),
                Out: T.Buffer((1,), "float32"),
            ):
                with T.Kernel(T.ceildiv(N, BLOCK), threads=BLOCK) as bx:
                    tx = T.get_thread_binding(0)
                    accum = T.alloc_local((1,), "float32")
                    reduced = T.alloc_local((1,), "float32")
                    T.clear(accum)
                    idx = bx * BLOCK + tx
                    if idx < N:
                        accum[0] += A[idx]
                    with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                    ):
                        T.evaluate(
                            T.tvm_thread_allreduce(
                                T.uint32(1),
                                accum[0],
                                True,
                                reduced[0],
                                tx,
                                dtype="handle",
                            )
                        )
                    # cross-block accumulation via atomic; Out must be zeroed beforehand
                    if tx == 0:
                        T.atomic_add(Out[0], reduced[0])

            return main

        return kernel

    def __call__(self, A: torch.Tensor, Out: torch.Tensor) -> None:
        assert A.is_cuda and Out.is_cuda
        assert A.shape == (self.n,) and Out.shape == (1,)
        Out.zero_()
        self._kernel(self.block_size)(A, Out)


@register_kernel("reduce", "tilelang_v1")
class ReduceV1TileLangKernel(BaseKernel):
    def __init__(self, n: int):
        self.n = n
        self.block_size = 256
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        N = self.n

        @tilelang.jit(out_idx=None)
        def kernel(BLOCK: int):
            NUM_WARP = BLOCK // 32

            @T.prim_func
            def main(
                A: T.Buffer((N,), "float32"),
                Out: T.Buffer((1,), "float32"),
            ):
                with T.Kernel(T.ceildiv(N, BLOCK), threads=BLOCK) as bx:
                    tx = T.get_thread_binding(0)
                    accm = T.alloc_local((1,), T.float)
                    T.clear(accm)
                    s_warp = T.alloc_shared((T.ceildiv(BLOCK, 32),), T.float)

                    idx = bx * BLOCK + tx
                    warp_id = tx // 32
                    lane_id = tx % 32
                    if idx < N:
                        accm[0] = A[idx]
                    # syncthread
                    s_warp[warp_id] = T.warp_reduce_sum(accm[0])
                    if warp_id == 0:
                        if lane_id < NUM_WARP:
                            accm[0] = s_warp[lane_id]
                        else:
                            accm[0] = 0

                        accm[0] = T.warp_reduce_sum(accm[0])
                        if lane_id == 0:
                            T.atomic_add(Out[0], accm[0])

            return main

        return kernel

    def __call__(self, A: torch.Tensor, Out: torch.Tensor) -> None:
        assert A.is_cuda and Out.is_cuda
        assert A.shape == (self.n,) and Out.shape == (1,)
        Out.zero_()
        self._kernel(self.block_size)(A, Out)
