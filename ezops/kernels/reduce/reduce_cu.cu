#include <tvm/ffi/tvm_ffi.h>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Sum reduction: Out[0] = sum(A). Each block reduces its chunk via warp
// shuffles and accumulates into Out with a single atomicAdd per block.
__global__ void reduce_kernel(const float* __restrict__ A, float* __restrict__ Out, int n) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * BLOCK_SIZE + tid;
  float val = (idx < n) ? __ldg(A + idx) : 0.0f;

  val = warp_reduce_sum(val);

  __shared__ float s_warp[NUM_WARPS];
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  if (lane_id == 0) s_warp[warp_id] = val;
  __syncthreads();

  if (warp_id == 0) {
    val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane_id == 0) atomicAdd(Out, val);
  }
}

void reduce_cu(tvm::ffi::TensorView A, tvm::ffi::TensorView Out) {
  int64_t n = A.size(0);
  int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  DLDevice dev = A.device();
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  // Out must be zeroed before the atomic accumulation
  cudaMemsetAsync(Out.data_ptr(), 0, sizeof(float), stream);
  reduce_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
      static_cast<const float*>(A.data_ptr()),
      static_cast<float*>(Out.data_ptr()),
      static_cast<int>(n));
}
