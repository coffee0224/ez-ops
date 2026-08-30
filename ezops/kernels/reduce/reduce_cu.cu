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



__device__ void warpReduce(volatile float* cache,int tid){
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}

__global__ void reduce_kernel_v4(const float* __restrict__ A, float* __restrict__ Out, int n) {
  __shared__ float sdata[BLOCK_SIZE];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + tid;

  if (i < n) {
    sdata[tid] = A[i] + A[i + BLOCK_SIZE];
  } else {
    sdata[tid] = 0;
  }
  __syncthreads();
  
  for(int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid+s];
    }
    __syncthreads();
  }

  if(tid < 32) {
    warpReduce(sdata,tid);
  }
  if (tid == 0) {
    atomicAdd(Out, sdata[0]);
  }
}



// Vectorized loads (float4) + per-thread register accumulation: each thread
// sums UNROLL*4 elements in registers before writing shared memory once.
constexpr int UNROLL = 4;

__global__ void reduce_kernel_v5(const float* __restrict__ A, float* __restrict__ Out, int n) {
  __shared__ float sdata[BLOCK_SIZE];

  int tid = threadIdx.x;
  int n_vec = n / 4;  // number of full float4's, tail handled scalar below
  const float4* A4 = reinterpret_cast<const float4*>(A);

  int vidx = blockIdx.x * (BLOCK_SIZE * UNROLL) + tid;
  float val = 0.0f;
  #pragma unroll
  for (int j = 0; j < UNROLL; ++j) {
    int idx = vidx + j * BLOCK_SIZE;
    if (idx < n_vec) {
      float4 v = __ldg(A4 + idx);
      val += (v.x + v.y) + (v.z + v.w);
    }
  }
  // scalar tail (n % 4 elements), covered once by block 0
  int rem = n - n_vec * 4;
  if (blockIdx.x == 0 && tid < rem) {
    val += A[n_vec * 4 + tid];
  }

  sdata[tid] = val;
  __syncthreads();

  for (int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    warpReduce(sdata, tid);
  }
  if (tid == 0) {
    atomicAdd(Out, sdata[0]);
  }
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}
// Sum reduction: Out[0] = sum(A). Each block reduces its chunk via warp
// shuffles and accumulates into Out with a single atomicAdd per block.
__global__ void reduce_kernel_v6(const float* __restrict__ A, float* __restrict__ Out, int n) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * BLOCK_SIZE * 2 + tid;
  float val1 = (idx < n) ? __ldg(A + idx) : 0.0f;
  float val2 = (idx < n) ? __ldg(A + idx + BLOCK_SIZE) : 0.0f; 

  val1 = warp_reduce_sum(val1);
  val2 = warp_reduce_sum(val2);

  __shared__ float s_warp[NUM_WARPS * 2];
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  if (lane_id == 0) {
    s_warp[warp_id] = val1;
    s_warp[warp_id + NUM_WARPS] = val2;
  }

  __syncthreads();

  if (warp_id == 0) {
    val1 = (lane_id < NUM_WARPS * 2) ? s_warp[lane_id] : 0.0f;
    val1 = warp_reduce_sum(val1);
    if (lane_id == 0) atomicAdd(Out, val1);
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

  // grid = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
  // reduce_kernel_v3<<<grid, BLOCK_SIZE, 0, stream>>>(
  //     static_cast<const float*>(A.data_ptr()),
  //     static_cast<float*>(Out.data_ptr()),
  //     static_cast<int>(n));

  // grid = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
  // reduce_kernel_v4<<<grid, BLOCK_SIZE, 0, stream>>>(
  //     static_cast<const float*>(A.data_ptr()),
  //     static_cast<float*>(Out.data_ptr()),
  //     static_cast<int>(n));

  // int n_vec = static_cast<int>(n) / 4;
  // grid = (n_vec + BLOCK_SIZE * UNROLL - 1) / (BLOCK_SIZE * UNROLL);
  // if (grid < 1) grid = 1;  // tail-only launches still need one block
  // reduce_kernel_v5<<<grid, BLOCK_SIZE, 0, stream>>>(
  //     static_cast<const float*>(A.data_ptr()),
  //     static_cast<float*>(Out.data_ptr()),
  //     static_cast<int>(n));

  grid = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
  reduce_kernel_v6<<<grid, BLOCK_SIZE, 0, stream>>>(
      static_cast<const float*>(A.data_ptr()),
      static_cast<float*>(Out.data_ptr()),
      static_cast<int>(n));
}
