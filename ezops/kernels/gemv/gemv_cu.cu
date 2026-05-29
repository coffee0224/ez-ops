#include <tvm/ffi/tvm_ffi.h>
#include <cuda_bf16.h>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

// Shared memory bank conflict padding: 1 pad per 8 elements (stride 9)
#define SMEM_PAD_IDX(i) ((i) + (i) / 8)
#define SMEM_PAD_SIZE(n) ((n) + (n) / 8)

__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// GEMV kernel ported from ldg_matvec_qkv (decode_ldg.cu):
//   C[i] = sum_k A[k] * B[i, k]
// A: (K,), B: (N, K), C: (N,), all bfloat16.
__global__ void gemv_ldg_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int N,
    int K) {
  int block_id = blockIdx.x;
  int num_blocks = gridDim.x;
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  // Cache vector A in bank-conflict-padded shared memory
  extern __shared__ float s_A[];
  for (int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
    s_A[SMEM_PAD_IDX(i)] = __bfloat162float(__ldg(A + i));
  }
  __syncthreads();

  // Distribute rows across blocks, each warp computes one row at a time
  int rows_per_block = (N + num_blocks - 1) / num_blocks;
  int row_start = block_id * rows_per_block;
  int row_end = min(row_start + rows_per_block, N);

  for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
    int m = m_base + warp_id;
    if (m < row_end) {
      const __nv_bfloat16* B_row = B + m * K;
      float sum = 0.0f;
      #pragma unroll 4
      for (int k = lane_id * 8; k < K; k += WARP_SIZE * 8) {
        uint4 b_u4 = __ldg(reinterpret_cast<const uint4*>(B_row + k));
        __nv_bfloat16* b_ptr = reinterpret_cast<__nv_bfloat16*>(&b_u4);
        int pk = SMEM_PAD_IDX(k);
        sum += __bfloat162float(b_ptr[0]) * s_A[pk + 0] +
               __bfloat162float(b_ptr[1]) * s_A[pk + 1] +
               __bfloat162float(b_ptr[2]) * s_A[pk + 2] +
               __bfloat162float(b_ptr[3]) * s_A[pk + 3] +
               __bfloat162float(b_ptr[4]) * s_A[pk + 4] +
               __bfloat162float(b_ptr[5]) * s_A[pk + 5] +
               __bfloat162float(b_ptr[6]) * s_A[pk + 6] +
               __bfloat162float(b_ptr[7]) * s_A[pk + 7];
      }
      sum = warp_reduce_sum(sum);
      if (lane_id == 0) C[m] = __float2bfloat16(sum);
    }
  }
}

void gemv_cu(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
  int64_t K = A.size(0);
  int64_t N = B.size(0);

  int num_blocks = (N + NUM_WARPS - 1) / NUM_WARPS;
  int smem_size = SMEM_PAD_SIZE((int)K) * sizeof(float);

  DLDevice dev = A.device();
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  gemv_ldg_kernel<<<num_blocks, BLOCK_SIZE, smem_size, stream>>>(
      static_cast<const __nv_bfloat16*>(A.data_ptr()),
      static_cast<const __nv_bfloat16*>(B.data_ptr()),
      static_cast<__nv_bfloat16*>(C.data_ptr()),
      static_cast<int>(N),
      static_cast<int>(K));
}
