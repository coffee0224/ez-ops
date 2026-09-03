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

// Out[b, :] = X[b, :] * rsqrt(mean(X[b, :]^2) + eps); one block per row,
// strided two-pass over the row with a warp-shuffle + shared block reduction.
__global__ void rmsnorm_kernel(const float* __restrict__ X, float* __restrict__ Out, int dim, float eps) {
  int row = blockIdx.x;
  const float* x = X + static_cast<int64_t>(row) * dim;
  float* out = Out + static_cast<int64_t>(row) * dim;

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  float acc = 0.0f;
  for (int i = tid; i < dim; i += BLOCK_SIZE) {
    float v = __ldg(x + i);
    acc += v * v;
  }

  __shared__ float s_warp[NUM_WARPS];
  __shared__ float s_scale;

  s_warp[warp_id] = warp_reduce_sum(acc);
  __syncthreads();
  if (warp_id == 0) {
    acc = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
    float total = warp_reduce_sum(acc);
    if (lane_id == 0) {
      s_scale = rsqrtf(total / dim + eps);
    }
  }
  __syncthreads();

  for (int i = tid; i < dim; i += BLOCK_SIZE) {
    out[i] = __ldg(x + i) * s_scale;
  }
}

void rmsnorm_cu(tvm::ffi::TensorView X, tvm::ffi::TensorView Out, double eps) {
  int64_t batch = X.size(0);
  int64_t dim = X.size(1);

  DLDevice dev = X.device();
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  rmsnorm_kernel<<<static_cast<int>(batch), BLOCK_SIZE, 0, stream>>>(
      static_cast<const float*>(X.data_ptr()),
      static_cast<float*>(Out.data_ptr()),
      static_cast<int>(dim),
      static_cast<float>(eps));
}
