#include <tvm/ffi/tvm_ffi.h>

// TODO: implement the CUDA kernel for gemv
__global__ void gemv_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K) {
  // TODO: kernel body
}

void gemv_cu(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t N = B.size(1);
  int block_size = 256;
  int grid = M;

  DLDevice dev = A.device();
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  gemv_kernel<<<grid, block_size, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(A.data_ptr()),
      static_cast<const __nv_bfloat16*>(B.data_ptr()),
      static_cast<__nv_bfloat16*>(C.data_ptr()),
      static_cast<int>(M),
      static_cast<int>(N),
      static_cast<int>(K));
}
