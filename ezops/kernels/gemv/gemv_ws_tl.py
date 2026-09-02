import os

os.environ.setdefault("TILELANG_CACHE_DIR", os.path.join(os.getcwd(), ".tilelang"))

import logging

import tilelang
import torch
from tilelang import PassConfigKey
from tilelang import language as T
from tilelang.carver.arch.driver.cuda_driver import get_num_sms

logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("gemv", "ws_gemv_tilelang")
class WsGemvTileLangKernel(BaseKernel):
    """Warp-specialized persistent GEMV in tilelang.

    Structural port of ezops/kernels/gemv/gemvws_cu.cu (ws_cuda):

      C[N] = A[K] @ B[N,K], A/B/C bf16, fp32 accumulation.

    Block layout (288 threads = 9 warps, persistent grid = num_sms):
      * warp 0  -- DMA/loader:
          - B tiles are streamed into a NUM_STAGES-deep smem ring with
            cp.async (16B chunks: per tile, 8 rows x TILE_K/256 column
            blocks, one coalesced 512B warp pass each), one
            commit_group + cp.async.mbarrier.arrive (T.cp_async_barrier_noinc)
            per tile, so the "data ready" mbarrier (count 32, one arrive per
            loader lane) fires when the async copies *complete*, not when
            the loader thread reaches the barrier.
          - before reusing a ring slot it waits on the "slot free" mbarrier
            (parity wait) that the compute warps arrive at after consuming.
      * warps 1..8 -- compute (one row per warp):
          - parity-wait on the "data ready" mbarrier of the slot, read the
            B tile row + the corresponding A segment from smem, fp32 dot
            product, warp_reduce_sum, lane-0-ordered store (all lanes of the
            warp write the same reduced value, ps_gemv precedent).
          - every compute thread (256) arrives at the "slot free" mbarrier,
            which guarantees each thread's smem reads are release-ordered
            before the loader may overwrite the slot.

    A is loaded once, whole, into smem by all 288 threads via T.copy,
    followed by a block-wide mbarrier (count 288) used as __syncthreads
    before the warps diverge into their roles.

    Persistent schedule: for rg = blockIdx; rg < total_row_groups;
    rg += num_blocks, one row group = 8 rows.  Ring slots are indexed by
    the block's *local* tile counter (it * num_tiles_k + kt, with it the
    persistent-loop iteration), keeping the pipeline primed across
    row-group boundaries while staying within per-block mbarriers.

    Synchronization is fully hand-written (no T.Pipelined) and
    TL_DISABLE_WARP_SPECIALIZED is set so the automatic warp-specialization
    pass does not re-shape the manually specialized structure.

    Constraints (asserted): K % TILE_K == 0, N % 8 == 0, K <= 16384
    (shared memory budget for s_A + the B ring); TILE_K must be a
    multiple of 256 (loader column-block width).

    Future PDL insertion points (not enabled, structure-only validation):
      * griddepcontrol.launch_dependents after the loader has issued its
        first cp.async groups (end of the first row-group issue loop);
      * griddepcontrol.wait before the compute warps' first mbarrier wait /
        first smem read of data written by this kernel's own prologue.
    """

    WARP_SIZE = 32
    COMPUTE_WARPS = 8
    DMA_WARPS = 1
    BLOCK_THREADS = (COMPUTE_WARPS + DMA_WARPS) * WARP_SIZE  # 288
    TILE_K = 512
    NUM_STAGES = 4

    def __init__(self, N: int, K: int):
        assert K % self.TILE_K == 0, f"K ({K}) must be a multiple of TILE_K ({self.TILE_K})"
        assert N % self.COMPUTE_WARPS == 0, f"N ({N}) must be a multiple of {self.COMPUTE_WARPS}"
        self.N = N
        self.K = K
        self.num_sms = get_num_sms()
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        N = self.N
        K = self.K
        num_sms = self.num_sms

        WARP_SIZE = self.WARP_SIZE
        COMPUTE_WARPS = self.COMPUTE_WARPS
        BLOCK_THREADS = self.BLOCK_THREADS
        TILE_K = self.TILE_K
        NUM_STAGES = self.NUM_STAGES

        # loader: 8 rows x TILE_K bf16; per (row, 256-col block) the 32 lanes
        # move one 16B chunk each (32 * 8 elems = 256 cols per pass)
        CP_CHUNK = 8
        COL_BLOCKS = TILE_K // (WARP_SIZE * CP_CHUNK)
        # compute: each lane consumes VEC elems per inner iteration
        VEC = 8
        COMPUTE_ITERS = TILE_K // (WARP_SIZE * VEC)
        TILE_ELEMS = COMPUTE_WARPS * TILE_K
        LANE_STRIDE = WARP_SIZE * VEC

        num_tiles_k = K // TILE_K
        total_row_groups = N // COMPUTE_WARPS
        num_iters = (total_row_groups + num_sms - 1) // num_sms

        @tilelang.jit(
            out_idx=None,
            # keep the automatic warp-specialization pass away from the
            # hand-written role split...
            pass_configs={
                PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
                # ...and stop the compiler from injecting __syncthreads()
                # between the loader's cp.async writes and the compute
                # warps' reads of s_B: that rendezvous serializes the whole
                # row-group (loader must finish issuing every K-tile before
                # any compute starts) and deadlocks the ring whenever
                # NUM_STAGES < K / TILE_K. All cross-thread smem sync here
                # is done manually with mbarriers.
                PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
            },
        )
        def kernel(
            dtype: T.dtype = T.bfloat16,
            accum_dtype: T.dtype = T.float,
        ):
            @T.prim_func
            def main(
                A: T.Buffer((K,), dtype),
                B: T.Buffer((N, K), dtype),
                C: T.Buffer((N,), dtype),
            ):
                with T.Kernel(num_sms, threads=BLOCK_THREADS) as block_id:
                    tx = T.get_thread_binding(0)
                    warp_id = tx // WARP_SIZE
                    lane = tx % WARP_SIZE
                    comp_idx = warp_id - 1  # row inside the row group

                    s_A = T.alloc_shared((K,), dtype)
                    s_B = T.alloc_shared((NUM_STAGES * TILE_ELEMS,), dtype)

                    # barrier 0: block-wide sync after the A load (count 288)
                    # full[s]:  B tile in ring slot s is ready (32 loader-lane
                    #           arrives fired by cp.async group completion)
                    # empty[s]: ring slot s is free again (256 compute threads)
                    sync_bar = T.alloc_barrier([BLOCK_THREADS])
                    full = T.alloc_barrier([WARP_SIZE] * NUM_STAGES)
                    empty = T.alloc_barrier([COMPUTE_WARPS * WARP_SIZE] * NUM_STAGES)

                    A_local = T.alloc_local((VEC,), dtype)
                    B_local = T.alloc_local((VEC,), dtype)
                    acc = T.alloc_local((1,), accum_dtype)

                    # ---- prologue: whole A into smem, block-wide sync ----
                    T.copy(A, s_A)
                    T.mbarrier_arrive(sync_bar[0])
                    T.mbarrier_wait_parity(sync_bar[0], 0)

                    # ---- persistent loop over row groups ----
                    for it in T.serial(num_iters):
                        rg = it * num_sms + block_id
                        if rg < total_row_groups:
                            row_base = rg * COMPUTE_WARPS
                            my_row = row_base + comp_idx

                            # ===== role: DMA / loader (warp 0) =====
                            # ring slots are indexed by this block's LOCAL
                            # tile counter (mbarriers live in per-block smem;
                            # the global rg would refer to another block's
                            # slots and deadlock)
                            if warp_id == 0:
                                for kt in T.serial(num_tiles_k):
                                    lit = it * num_tiles_k + kt
                                    s = lit % NUM_STAGES
                                    phase = lit // NUM_STAGES
                                    if phase >= 1:
                                        # slot reuse: wait until compute
                                        # finished the previous use
                                        T.mbarrier_wait_parity(empty[s], (phase - 1) % 2)
                                    for r in T.serial(COMPUTE_WARPS):
                                        for cb in T.serial(COL_BLOCKS):
                                            T.ptx_cp_async(
                                                T.tvm_access_ptr(
                                                    T.type_annotation(dtype),
                                                    s_B.data,
                                                    s * TILE_ELEMS
                                                    + r * TILE_K
                                                    + cb * (WARP_SIZE * CP_CHUNK)
                                                    + lane * CP_CHUNK,
                                                    CP_CHUNK,
                                                    2,
                                                ),
                                                T.tvm_access_ptr(
                                                    T.type_annotation(dtype),
                                                    B.data,
                                                    (row_base + r) * K
                                                    + kt * TILE_K
                                                    + cb * (WARP_SIZE * CP_CHUNK)
                                                    + lane * CP_CHUNK,
                                                    CP_CHUNK,
                                                    1,
                                                ),
                                                CP_CHUNK,
                                            )
                                    T.ptx_commit_group()
                                    # arrive on full[s] when this lane's
                                    # committed cp.async group completes
                                    T.cp_async_barrier_noinc(full[s])

                            # ===== role: compute (warps 1..8) =====
                            if warp_id >= 1:
                                T.clear(acc)
                                for kt in T.serial(num_tiles_k):
                                    lit = it * num_tiles_k + kt
                                    s = lit % NUM_STAGES
                                    phase = lit // NUM_STAGES
                                    T.mbarrier_wait_parity(full[s], phase % 2)
                                    base_b = s * TILE_ELEMS + comp_idx * TILE_K
                                    base_a = kt * TILE_K
                                    for j in T.serial(COMPUTE_ITERS):
                                        for e in T.vectorized(VEC):
                                            A_local[e] = s_A[base_a + j * LANE_STRIDE + lane * VEC + e]
                                        for e in T.vectorized(VEC):
                                            B_local[e] = s_B[base_b + j * LANE_STRIDE + lane * VEC + e]
                                        for e in T.serial(VEC):
                                            acc[0] += (
                                                A_local[e].astype(accum_dtype)
                                                * B_local[e].astype(accum_dtype)
                                            )
                                    T.mbarrier_arrive(empty[s])
                                if my_row < N:
                                    C[my_row] = T.warp_reduce_sum(acc[0])

            return main

        return kernel

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        self._kernel()(A, B, C)
