import os

os.environ.setdefault("TILELANG_CACHE_DIR", os.path.join(os.getcwd(), ".tilelang"))

import logging

import tilelang
import torch
from tilelang import language as T

logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)

from ..base_kernel import BaseKernel
from ...registry import register_kernel


# Tile config — picked to mirror the CUDA version's structure (BM=BN=64) but
# scaled up to tilelang's tensor-core defaults (BM=BN=128, BK=32). Each block
# computes a 128x128 output tile.
_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_K = 32
_NUM_STAGES = 3
_THREADS = 128


def _make_gemm_jit(use_pdl: bool):
    """Build a tilelang JIT function for a single GEMM C[M,N] = A[M,K] @ B[K,N].

    A/B/C buffers are FP32 (matches the op interface and the CUDA baseline).
    Internally the K-loop loads tiles into BF16 shared memory — T.copy
    auto-casts — so we can hit Blackwell's BF16 tensor cores. Accumulator
    and output are FP32.
    """

    @tilelang.jit(out_idx=None)
    def gemm(
        M: int,
        K: int,
        N: int,
        BLOCK_M: int = _BLOCK_M,
        BLOCK_N: int = _BLOCK_N,
        BLOCK_K: int = _BLOCK_K,
        num_stages: int = _NUM_STAGES,
        threads: int = _THREADS,
    ):
        io_dtype = T.float32       # buffer dtype (FP32, matches op interface)
        compute_dtype = T.float16  # sm_120a has no FP32 tensor core, use FP16
        accum_dtype = T.float32

        @T.prim_func
        def main(
            A: T.Buffer((M, K), io_dtype),
            B: T.Buffer((K, N), io_dtype),
            C: T.Buffer((M, N), io_dtype),
        ):
            with T.Kernel(
                T.ceildiv(N, BLOCK_N),
                T.ceildiv(M, BLOCK_M),
                threads=threads,
            ) as (bx, by):
                # BF16 shared tiles so T.gemm hits BF16 tensor cores.
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), compute_dtype)
                B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), compute_dtype)
                C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

                T.clear(C_local)

                # PDL: wait for the previous kernel in the stream to commit
                # its outputs. For FC1 (no predecessor) this is a no-op; for
                # FC2 it blocks until FC1's grid-ending membar commits y.
                # Side effect: tilelang sees pdl_sync and adds the
                # ProgrammaticStreamSerialization launch attribute on host.
                if use_pdl:
                    T.pdl_sync()

                K_blocks = T.ceildiv(K, BLOCK_K)
                trigger_at = K_blocks // 2

                for ko in T.Pipelined(K_blocks, num_stages=num_stages):
                    # T.copy auto-casts FP32 global -> BF16 shared.
                    T.copy(A[by * BLOCK_M, ko * BLOCK_K], A_shared)
                    T.copy(B[ko * BLOCK_K, bx * BLOCK_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                    # PDL: midway through the K loop, signal the hardware to
                    # launch the next kernel. Its prolog (smem alloc, T.copy
                    # setup) overlaps with the rest of our mainloop + epilogue
                    # + grid-ending membar.
                    if use_pdl:
                        with T.If(ko + 1 == trigger_at):
                            with T.Then():
                                T.pdl_trigger()

                # PDL: fallback for very short K (trigger_at == 0 means the
                # in-loop branch never matched). The hardware dedups repeat
                # triggers, so this is safe even when the in-loop one fired.
                if use_pdl:
                    T.pdl_trigger()

                T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])

        return main

    return gemm


class _PdlGemmTileLangBase(BaseKernel):
    USE_PDL = False

    def __init__(self, M: int, K: int, N: int, P: int):
        self.M, self.K, self.N, self.P = M, K, N, P
        gemm_jit = _make_gemm_jit(self.USE_PDL)
        # Two pre-compiled GEMM kernels: FC1 maps (M,K) -> (M,N);
        # FC2 maps (M,N) -> (M,P).
        self._fc1 = gemm_jit(M=M, K=K, N=N)
        self._fc2 = gemm_jit(M=M, K=N, N=P)

    def __call__(
        self,
        x: torch.Tensor,
        W1: torch.Tensor,
        W2: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> None:
        # y = x @ W1
        self._fc1(x, W1, y)
        # z = y @ W2
        self._fc2(y, W2, z)


@register_kernel("pdl_gemm", "tl")
class PdlGemmTileLangKernel(_PdlGemmTileLangBase):
    """Tilelang GEMM, no PDL. Two serial kernel launches on the same stream."""

    USE_PDL = False


@register_kernel("pdl_gemm", "tl_pdl")
class PdlGemmTileLangPdlKernel(_PdlGemmTileLangBase):
    """Tilelang GEMM with PDL primitives. FC1 triggers FC2 mid-mainloop so
    FC2's prolog overlaps with FC1's epilogue + grid-ending membar."""

    USE_PDL = True
