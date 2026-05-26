import tempfile
from pathlib import Path

import torch

from ..base_kernel import BaseKernel
from ...registry import register_kernel

CUDA_KERNEL_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

void launch_vector_add(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t n, int64_t block_size) {
    int grid = (n + block_size - 1) / block_size;
    vector_add_kernel<<<grid, block_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_vector_add, "vector_add launch");
}
"""


@register_kernel("vector_add", "cuda")
class VectorAddCudaKernel(BaseKernel):
    def __init__(self, n: int):
        self.n = n
        self.block_size = 256
        self._compiled = self._compile()

    def _compile(self):
        from torch.utils.cpp_extension import load

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = Path(tmpdir) / "vector_add.cu"
            src_path.write_text(CUDA_KERNEL_SRC)
            return load(name="vector_add_cuda", sources=[str(src_path)], extra_cuda_cflags=["-O2"])

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.is_cuda and B.is_cuda and C.is_cuda
        self._compiled.launch(A, B, C, self.n, self.block_size)
