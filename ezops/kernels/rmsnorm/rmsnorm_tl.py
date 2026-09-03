import logging

import tilelang
import torch
from tilelang import language as T

# tilelang's KernelCache warns "consider using @tilelang.jit" on cache hits,
# even when we already use @tilelang.jit. Suppress the misleading warning.
logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("rmsnorm", "tilelang")
class RmsNormTileLangKernel(BaseKernel):
    def __init__(self, batch_size: int, dim: int, eps: float = 1e-6):
        self.batch_size = batch_size
        self.dim = dim
        self.eps = eps
        self.block_size = 256
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        B, D, eps = self.batch_size, self.dim, self.eps

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
                    acc = T.alloc_local((1,), "float32")
                    total = T.alloc_local((1,), "float32")
                    s_warp = T.alloc_shared((NUM_WARP,), "float32")
                    s_scale = T.alloc_shared((1,), "float32")

                    # pass 1: strided sum of squares, one block per row
                    T.clear(acc)
                    for kb in T.serial(T.ceildiv(D, BLOCK)):
                        idx = kb * BLOCK + tx
                        if idx < D:
                            v = X[bx, idx]
                            acc[0] += v * v
                    s_warp[warp_id] = T.warp_reduce_sum(acc[0])
                    if warp_id == 0:
                        if lane_id < NUM_WARP:
                            acc[0] = s_warp[lane_id]
                        else:
                            acc[0] = 0.0
                        total[0] = T.warp_reduce_sum(acc[0])
                        if lane_id == 0:
                            s_scale[0] = T.rsqrt(total[0] / D + eps)
                    # pass 2: normalize
                    for kb in T.serial(T.ceildiv(D, BLOCK)):
                        idx = kb * BLOCK + tx
                        if idx < D:
                            Out[bx, idx] = X[bx, idx] * s_scale[0]

            return main

        return kernel

    def __call__(self, X: torch.Tensor, Out: torch.Tensor) -> None:
        assert X.is_cuda and Out.is_cuda
        assert X.shape == (self.batch_size, self.dim) and Out.shape == X.shape
        self._kernel(self.block_size)(X, Out)
