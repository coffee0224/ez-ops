from pathlib import Path

import torch
from tvm_ffi import cpp

from ..base_kernel import BaseKernel
from ...registry import register_kernel

_CU_SRC = Path(__file__).parent / "pdl_gemm_cu.cu"

_COMMON_CFLAGS = [
    "-O3",
    "--generate-line-info",
    "-arch=sm_120a",
]


@register_kernel("pdl_gemm", "cuda")
class PdlGemmCudaKernel(BaseKernel):
    """Baseline: two GEMM kernels launched back-to-back on the same stream.
    No PDL. The second kernel waits for the first's grid-ending membar."""

    def __init__(self, M: int, K: int, N: int, P: int):
        self.M, self.K, self.N, self.P = M, K, N, P
        self._mod = cpp.load_inline(
            name="pdl_gemm_cuda",
            cuda_sources=_CU_SRC.read_text(),
            functions=["pdl_gemm_baseline_cu", "pdl_gemm_pdl_cu"],
            extra_cuda_cflags=_COMMON_CFLAGS,
        )

    def __call__(
        self,
        x: torch.Tensor,
        W1: torch.Tensor,
        W2: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> None:
        self._mod.pdl_gemm_baseline_cu(x, W1, W2, y, z)


@register_kernel("pdl_gemm", "cuda_pdl")
class PdlGemmPdlCudaKernel(BaseKernel):
    """PDL: same GEMM kernel, but launched with
    cudaLaunchAttributeProgrammaticStreamSerialization and emits
    griddepcontrol PTX so FC2's prolog overlaps with FC1's epilogue."""

    def __init__(self, M: int, K: int, N: int, P: int):
        self.M, self.K, self.N, self.P = M, K, N, P
        # Reuse the compiled module (cached by name in tvm_ffi.cpp).
        self._mod = cpp.load_inline(
            name="pdl_gemm_cuda",
            cuda_sources=_CU_SRC.read_text(),
            functions=["pdl_gemm_baseline_cu", "pdl_gemm_pdl_cu"],
            extra_cuda_cflags=_COMMON_CFLAGS,
        )

    def __call__(
        self,
        x: torch.Tensor,
        W1: torch.Tensor,
        W2: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> None:
        self._mod.pdl_gemm_pdl_cu(x, W1, W2, y, z)
