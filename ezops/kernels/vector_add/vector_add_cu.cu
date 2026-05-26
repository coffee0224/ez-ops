#include <tvm/ffi/tvm_ffi.h>

__global__ void vector_add_kernel(const float* A, const float* B, float* C, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    C[idx] = A[idx] + B[idx];
  }
}

void vector_add_cu(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
  int64_t n = A.size(0);
  int block_size = 256;
  int grid = (n + block_size - 1) / block_size;

  DLDevice dev = A.device();
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  vector_add_kernel<<<grid, block_size, 0, stream>>>(
      static_cast<const float*>(A.data_ptr()),
      static_cast<const float*>(B.data_ptr()),
      static_cast<float*>(C.data_ptr()),
      static_cast<int>(n));
}
