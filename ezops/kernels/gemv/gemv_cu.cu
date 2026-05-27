#include <tvm/ffi/tvm_ffi.h>

// TODO: implement the CUDA kernel for gemv
__global__ void gemv_kernel(const float* A, const float* x, float* y, int M, int N) {
  // TODO: kernel body
}

void gemv_cu(tvm::ffi::TensorView A, tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  int64_t M = A.size(0);
  int64_t N = A.size(1);
  int block_size = 256;
  int grid = M;

  DLDevice dev = A.device();
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  gemv_kernel<<<grid, block_size, 0, stream>>>(
      static_cast<const float*>(A.data_ptr()),
      static_cast<const float*>(x.data_ptr()),
      static_cast<float*>(y.data_ptr()),
      static_cast<int>(M),
      static_cast<int>(N));
}
