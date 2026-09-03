import logging

import tilelang
import torch
from tilelang import language as T

# tilelang's KernelCache warns "consider using @tilelang.jit" on cache hits,
# even when we already use @tilelang.jit. Suppress the misleading warning.
logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("softmax", "tilelang")
class SoftmaxTileLangKernel(BaseKernel):
    def __init__(self, batch_size: int, dim: int):
        self.batch_size = batch_size
        self.dim = dim
        self.block_size = 256
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        B, D = self.batch_size, self.dim

        @tilelang.jit(out_idx=None)
        def kernel(BLOCK: int):
            NUM_WARP = BLOCK // 32

            @T.prim_func
            def main(
                X: T.Buffer((B, D), "float32"),
                Out: T.Buffer((B, D), "float32"),
            ):
                with T.Kernel(B, threads=BLOCK) as bx:
                    tx = T.get_thread_binding(0)
                    warp_id = tx // 32
                    lane_id = tx % 32
                    m = T.alloc_local((1,), "float32")
                    s = T.alloc_local((1,), "float32")
                    s_warp = T.alloc_shared((NUM_WARP,), "float32")
                    s_max = T.alloc_shared((1,), "float32")
                    s_sum = T.alloc_shared((1,), "float32")

                    # pass 1: strided row max
                    m[0] = T.min_value("float32")
                    for kb in T.serial(T.ceildiv(D, BLOCK)):
                        idx = kb * BLOCK + tx
                        if idx < D:
                            m[0] = T.max(m[0], X[bx, idx])
                    s_warp[warp_id] = T.warp_reduce_max(m[0])
                    if warp_id == 0:
                        if lane_id < NUM_WARP:
                            m[0] = s_warp[lane_id]
                        else:
                            m[0] = T.min_value("float32")
                        m[0] = T.warp_reduce_max(m[0])
                        if lane_id == 0:
                            s_max[0] = m[0]
                    # pass 2: sum of exp(x - max)
                    T.clear(s)
                    for kb in T.serial(T.ceildiv(D, BLOCK)):
                        idx = kb * BLOCK + tx
                        if idx < D:
                            s[0] += T.exp(X[bx, idx] - s_max[0])
                    s_warp[warp_id] = T.warp_reduce_sum(s[0])
                    if warp_id == 0:
                        if lane_id < NUM_WARP:
                            s[0] = s_warp[lane_id]
                        else:
                            s[0] = 0.0
                        s[0] = T.warp_reduce_sum(s[0])
                        if lane_id == 0:
                            s_sum[0] = s[0]
                    # pass 3: normalize
                    for kb in T.serial(T.ceildiv(D, BLOCK)):
                        idx = kb * BLOCK + tx
                        if idx < D:
                            Out[bx, idx] = T.exp(X[bx, idx] - s_max[0]) / s_sum[0]

            return main

        return kernel

    def __call__(self, X: torch.Tensor, Out: torch.Tensor) -> None:
        assert X.is_cuda and Out.is_cuda
        assert X.shape == (self.batch_size, self.dim) and Out.shape == X.shape
        self._kernel(self.block_size)(X, Out)


@register_kernel("softmax", "tilelang_v1")
class SoftmaxV1TileLangKernel(BaseKernel):
    def __init__(self, batch_size: int, dim: int):
        self.batch_size = batch_size
        self.dim = dim
        self.block_size = 256
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        B, D = self.batch_size, self.dim

        @tilelang.jit(out_idx=None)
        def kernel(BLOCK: int):
            NUM_WARP = BLOCK // 32
            NUM_CHUNK = (D + BLOCK - 1) // BLOCK

            @T.prim_func
            def main(
                X: T.Buffer((B, D), "float32"),
                Out: T.Buffer((B, D), "float32"),
            ):
                with T.Kernel(B, threads=BLOCK) as bx:
                    tx = T.get_thread_binding(0)
                    warp_id = tx // 32
                    lane_id = tx % 32

                    x_cache = T.alloc_local((NUM_CHUNK,), T.float)
                    m = T.alloc_local((1,), T.float)
                    s = T.alloc_local((1,), T.float)
                    m_red = T.alloc_local((1,), T.float)
                    s_warp = T.alloc_shared((NUM_WARP,), T.float)
                    s_max = T.alloc_shared((1,), T.float)
                    s_sum = T.alloc_shared((1,), T.float)

                    # pass 1 (online): read X once into registers while
                    # maintaining a running max and a rescaled running
                    # sum of exp(x - m)
                    m[0] = T.min_value("float32")
                    T.clear(s)
                    for k in T.serial(NUM_CHUNK):
                        idx = k * BLOCK + tx
                        if idx < D:
                            v = X[bx, idx]
                            x_cache[k] = v
                            m_new = T.max(m[0], v)
                            s[0] = s[0] * T.exp(m[0] - m_new) + T.exp(v - m_new)
                            m[0] = m_new

                    # block-reduce the per-thread running max (m_red is
                    # reduction scratch; m must survive for the rescale)
                    s_warp[warp_id] = T.warp_reduce_max(m[0])
                    if warp_id == 0:
                        if lane_id < NUM_WARP:
                            m_red[0] = s_warp[lane_id]
                        else:
                            m_red[0] = T.min_value("float32")
                        m_red[0] = T.warp_reduce_max(m_red[0])
                        if lane_id == 0:
                            s_max[0] = m_red[0]

                    # rescale per-thread sums to the block max, then reduce
                    s[0] = s[0] * T.exp(m[0] - s_max[0])
                    s_warp[warp_id] = T.warp_reduce_sum(s[0])
                    if warp_id == 0:
                        if lane_id < NUM_WARP:
                            s[0] = s_warp[lane_id]
                        else:
                            s[0] = 0.0
                        s[0] = T.warp_reduce_sum(s[0])
                        if lane_id == 0:
                            s_sum[0] = s[0]

                    # pass 2: normalize straight from the register cache
                    for k in T.serial(NUM_CHUNK):
                        idx = k * BLOCK + tx
                        if idx < D:
                            Out[bx, idx] = T.exp(x_cache[k] - s_max[0]) / s_sum[0]

            return main

        return kernel

    def __call__(self, X: torch.Tensor, Out: torch.Tensor) -> None:
        assert X.is_cuda and Out.is_cuda
        assert X.shape == (self.batch_size, self.dim) and Out.shape == X.shape
        self._kernel(self.block_size)(X, Out)
