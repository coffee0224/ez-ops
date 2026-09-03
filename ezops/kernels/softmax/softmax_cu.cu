#include <tvm/ffi/tvm_ffi.h>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// Out[b, :] = exp(X[b, :] - max) / sum(exp(X[b, :] - max)); one block per row,
// three strided passes (max, sum of exp, normalize) with warp-shuffle block
// reductions broadcast through shared memory.
__global__ void softmax_kernel(const float* __restrict__ X, float* __restrict__ Out, int dim) {
  int row = blockIdx.x;
  const float* x = X + static_cast<int64_t>(row) * dim;
  float* out = Out + static_cast<int64_t>(row) * dim;

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  __shared__ float s_warp[NUM_WARPS];
  __shared__ float s_max;
  __shared__ float s_sum;

  // pass 1: row max
  float m = -INFINITY;
  for (int i = tid; i < dim; i += BLOCK_SIZE) {
    m = fmaxf(m, __ldg(x + i));
  }
  s_warp[warp_id] = warp_reduce_max(m);
  __syncthreads();
  if (warp_id == 0) {
    m = (lane_id < NUM_WARPS) ? s_warp[lane_id] : -INFINITY;
    m = warp_reduce_max(m);
    if (lane_id == 0) {
      s_max = m;
    }
  }
  __syncthreads();

  // pass 2: sum of exp(x - max)
  float s = 0.0f;
  for (int i = tid; i < dim; i += BLOCK_SIZE) {
    s += expf(__ldg(x + i) - s_max);
  }
  s_warp[warp_id] = warp_reduce_sum(s);
  __syncthreads();
  if (warp_id == 0) {
    s = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
    s = warp_reduce_sum(s);
    if (lane_id == 0) {
      s_sum = s;
    }
  }
  __syncthreads();

  // pass 3: normalize
  float inv = 1.0f / s_sum;
  for (int i = tid; i < dim; i += BLOCK_SIZE) {
    out[i] = expf(__ldg(x + i) - s_max) * inv;
  }
}

void softmax_cu(tvm::ffi::TensorView X, tvm::ffi::TensorView Out) {
  int64_t batch = X.size(0);
  int64_t dim = X.size(1);

  DLDevice dev = X.device();
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  softmax_kernel<<<static_cast<int>(batch), BLOCK_SIZE, 0, stream>>>(
      static_cast<const float*>(X.data_ptr()),
      static_cast<float*>(Out.data_ptr()),
      static_cast<int>(dim));
}
