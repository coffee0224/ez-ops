#include <tvm/ffi/tvm_ffi.h>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;


// baseline
__global__ void reduce_kernel_v0(const float* __restrict__ A, float* __restrict__ Out, int n) {
  __shared__ float sdata[BLOCK_SIZE];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  if (i < n) {
    sdata[tid] = A[i];
  } else {
    sdata[tid] = 0;
  }

  __syncthreads();
  for(int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {
      sdata[tid] += sdata[tid+s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(Out, sdata[0]);
  }
}

// 消除 warp_divergence

__global__ void reduce_kernel_v1(const float* __restrict__ A, float* __restrict__ Out, int n) {
  __shared__ float sdata[BLOCK_SIZE];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  if (i < n) {
    sdata[tid] = A[i];
  } else {
    sdata[tid] = 0;
  }
  __syncthreads();
  
  for(int s = 1; s < blockDim.x; s <<= 1) {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
      sdata[index] += sdata[index+s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(Out, sdata[0]);
  }
}

__global__ void reduce_kernel_v2(const float* __restrict__ A, float* __restrict__ Out, int n) {
  __shared__ float sdata[BLOCK_SIZE];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  if (i < n) {
    sdata[tid] = A[i];
  } else {
    sdata[tid] = 0;
  }
  __syncthreads();
  
  for(int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid+s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(Out, sdata[0]);
  }
}

__global__ void reduce_kernel_v3(const float* __restrict__ A, float* __restrict__ Out, int n) {
  __shared__ float sdata[BLOCK_SIZE];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + tid;

  if (i < n) {
    sdata[tid] = A[i] + A[i + BLOCK_SIZE];
  } else {
    sdata[tid] = 0;
  }
  __syncthreads();
  
  for(int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid+s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(Out, sdata[0]);
  }
}


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
  // reduce_kernel_v0<<<grid, BLOCK_SIZE, 0, stream>>>(
  //     static_cast<const float*>(A.data_ptr()),
  //     static_cast<float*>(Out.data_ptr()),
  //     static_cast<int>(n));

  // reduce_kernel_v1<<<grid, BLOCK_SIZE, 0, stream>>>(
  //     static_cast<const float*>(A.data_ptr()),
  //     static_cast<float*>(Out.data_ptr()),
  //     static_cast<int>(n));

  // reduce_kernel_v2<<<grid, BLOCK_SIZE, 0, stream>>>(
  //     static_cast<const float*>(A.data_ptr()),
  //     static_cast<float*>(Out.data_ptr()),
  //     static_cast<int>(n));

  grid = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
  reduce_kernel_v3<<<grid, BLOCK_SIZE, 0, stream>>>(
      static_cast<const float*>(A.data_ptr()),
      static_cast<float*>(Out.data_ptr()),
      static_cast<int>(n));

  // reduce_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
  //     static_cast<const float*>(A.data_ptr()),
  //     static_cast<float*>(Out.data_ptr()),
  //     static_cast<int>(n));
}
