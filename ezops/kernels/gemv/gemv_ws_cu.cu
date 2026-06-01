#include <tvm/ffi/tvm_ffi.h>
#include <cuda.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 288;  // 9 warps: 1 DMA + 8 compute
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
constexpr int DMA_WARPS = 1;
constexpr int COMPUTE_WARPS = NUM_WARPS - DMA_WARPS;  // 8
constexpr int TILE_K = 256;
constexpr int NUM_STAGES = 2;
constexpr int CP16B_ELEMS = 8;

// ---------------------------------------------------------------------------
// cp.async.cg helpers (for A loading)
// ---------------------------------------------------------------------------
#define CP_ASYNC_CG(dst_smem_32b, src_global_ptr)                                              \
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst_smem_32b),     \
               "l"(src_global_ptr))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

__device__ __forceinline__ void cp_async_wait_group_0() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

// ---------------------------------------------------------------------------
// Warp reduction
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// ---------------------------------------------------------------------------
// mbarrier helpers (SM90+)
// ---------------------------------------------------------------------------
typedef uint64_t mbarrier_t;

__device__ __forceinline__ void mbarrier_init(mbarrier_t* mb, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
               ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(mb))),
                 "r"(count));
}

__device__ __forceinline__ void mbarrier_expect_tx(mbarrier_t* mb, uint32_t tx_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
               ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(mb))),
                 "r"(tx_bytes));
}

__device__ __forceinline__ void mbarrier_wait(mbarrier_t* mb, uint32_t phase) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mb));
  int done = 0;
  while (!done) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t"
        "}\n\t"
        : "=r"(done)
        : "r"(smem_addr), "r"(phase));
  }
}

// ---------------------------------------------------------------------------
// TMA load helper (SM90+)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_bulk_tensor_2d(
    mbarrier_t* mb,
    const void* tmap,
    void* smem_ptr,
    int32_t coord_fast,
    int32_t coord_slow) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  uint32_t mb_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mb));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n"
      ::"r"(smem_addr),
        "l"(tmap),
        "r"(coord_fast),
        "r"(coord_slow),
        "r"(mb_addr)
      : "memory");
}

// ---------------------------------------------------------------------------
// Persistent warp-specialized GEMV kernel (TMA + mbarrier_wait)
//   C[N] = A[K] @ B[N,K]
//
// Warp 0: DMA producer (loads A via cp.async.cg, B via TMA)
// Warps 1-8: Compute consumers (each warp handles one row)
//
// Sync model:
//   - Compute warps call mbarrier_wait(mb[cur_buf]) to wait for TMA data.
//   - DMA thread fires TMA (expect_tx + issue) without waiting for completion.
//   - __syncthreads() at the bottom prevents DMA from getting too far ahead.
//   - Overlap: TMA for nxt_buf runs in background while compute works on cur_buf.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(288, 1)
gemv_ws_kernel(
    const __nv_bfloat16* __restrict__ A,
    __grid_constant__ const CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C,
    int N,
    int K,
    int num_tiles_k) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  bool is_dma = (warp_id == 0);
  int comp_idx = warp_id - 1;

  // ---- Shared memory layout ----
  extern __shared__ char smem_raw[];

  __nv_bfloat16* s_A = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  size_t s_A_bytes = (size_t)K * sizeof(__nv_bfloat16);
  size_t s_A_aligned = (s_A_bytes + 127) & ~(size_t)127;

  __nv_bfloat16* s_B_base = reinterpret_cast<__nv_bfloat16*>(smem_raw + s_A_aligned);
  __nv_bfloat16* s_B_buf[NUM_STAGES] = {
      s_B_base,
      s_B_base + COMPUTE_WARPS * TILE_K,
  };

  size_t s_B_bytes = NUM_STAGES * COMPUTE_WARPS * TILE_K * sizeof(__nv_bfloat16);
  size_t mb_offset = (s_A_aligned + s_B_bytes + 7) & ~(size_t)7;
  mbarrier_t* mb = reinterpret_cast<mbarrier_t*>(smem_raw + mb_offset);

  int num_blocks = gridDim.x;
  int total_row_groups = (N + COMPUTE_WARPS - 1) / COMPUTE_WARPS;

  // ---- Init mbarriers ----
  if (threadIdx.x == 0) {
    mbarrier_init(&mb[0], 1);
    mbarrier_init(&mb[1], 1);
  }
  __syncthreads();

  // ---- Phase 1: Load A into smem (cp.async.cg) ----
  if (is_dma) {
    int tile_16B = K / CP16B_ELEMS;
    for (int i = lane_id; i < tile_16B; i += WARP_SIZE) {
      uint32_t smem_addr = static_cast<uint32_t>(
          __cvta_generic_to_shared(s_A + i * CP16B_ELEMS));
      CP_ASYNC_CG(smem_addr, A + i * CP16B_ELEMS);
    }
    CP_ASYNC_COMMIT_GROUP();
    cp_async_wait_group_0();
    __syncwarp();
  }
  __syncthreads();

  // ---- Prologue: TMA load first B tile (fire and forget) ----
  if (threadIdx.x == 0 && num_tiles_k > 0 && blockIdx.x < total_row_groups) {
    int row_base = blockIdx.x * COMPUTE_WARPS;
    uint32_t tile_bytes = COMPUTE_WARPS * TILE_K * sizeof(__nv_bfloat16);
    mbarrier_expect_tx(&mb[0], tile_bytes);
    cp_async_bulk_tensor_2d(
        &mb[0], &tma_B, s_B_buf[0],
        /*coord_fast=*/0, /*coord_slow=*/row_base);
  }

  // ---- Phase 2: Persistent pipeline ----
  uint32_t compute_phase[2] = {0, 0};

  for (int rg = blockIdx.x; rg < total_row_groups; rg += num_blocks) {
    int row_base = rg * COMPUTE_WARPS;
    int my_row = row_base + comp_idx;
    float acc = 0.0f;

    for (int k_tile = 0; k_tile < num_tiles_k; k_tile++) {
      int cur_buf = k_tile & 1;
      int nxt_buf = cur_buf ^ 1;

      // Compute warps: wait for TMA data in cur_buf
      if (!is_dma) {
        mbarrier_wait(&mb[cur_buf], compute_phase[cur_buf]);
        compute_phase[cur_buf] ^= 1;
      }

      // Compute consumers: dot product with cur_buf
      if (!is_dma && my_row < N) {
        const __nv_bfloat16* B_row = s_B_buf[cur_buf] + comp_idx * TILE_K;
        int k_start = k_tile * TILE_K;

#pragma unroll 4
        for (int k = lane_id; k < TILE_K; k += WARP_SIZE) {
          acc += __bfloat162float(s_A[k_start + k]) * __bfloat162float(B_row[k]);
        }

        if (k_tile == num_tiles_k - 1) {
          acc = warp_reduce_sum(acc);
          if (lane_id == 0) {
            C[my_row] = __float2bfloat16(acc);
          }
          acc = 0.0f;
        }
      }

      // DMA producer: fire-and-forget TMA into nxt_buf
      if (threadIdx.x == 0) {
        int next_k_tile = k_tile + 1;
        int next_rg = rg;

        if (next_k_tile >= num_tiles_k) {
          next_k_tile = 0;
          next_rg = rg + num_blocks;
        }

        int next_row_base = next_rg * COMPUTE_WARPS;

        if (next_rg < total_row_groups) {
          uint32_t tile_bytes = COMPUTE_WARPS * TILE_K * sizeof(__nv_bfloat16);
          mbarrier_expect_tx(&mb[nxt_buf], tile_bytes);
          cp_async_bulk_tensor_2d(
              &mb[nxt_buf], &tma_B, s_B_buf[nxt_buf],
              static_cast<int32_t>(next_k_tile * TILE_K),
              static_cast<int32_t>(next_row_base));
        }
      }

      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------------
void gemv_ws_cu(
    tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
  int64_t K = A.size(0);
  int64_t N = B.size(0);

  int num_tiles_k = (K + TILE_K - 1) / TILE_K;

  DLDevice dev = A.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  int device_id = dev.device_id;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  int num_sms = prop.multiProcessorCount;

  // ---- Create TMA descriptor for B[N, K] ----
  CUtensorMap tma_B;
  uint64_t globalDim[2] = {(uint64_t)K, (uint64_t)N};
  uint64_t globalStrides[1] = {(uint64_t)(K * sizeof(__nv_bfloat16))};
  uint32_t boxDim[2] = {(uint32_t)TILE_K, (uint32_t)COMPUTE_WARPS};
  uint32_t elementStrides[2] = {1, 1};

  CUresult res = cuTensorMapEncodeTiled(
      &tma_B,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      2,
      B.data_ptr(),
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "[gemv_ws] cuTensorMapEncodeTiled failed: %d\n", (int)res);
    return;
  }

  // ---- Shared memory sizing ----
  size_t s_A_bytes = (size_t)K * sizeof(__nv_bfloat16);
  size_t s_A_aligned = (s_A_bytes + 127) & ~(size_t)127;
  size_t s_B_bytes = NUM_STAGES * COMPUTE_WARPS * TILE_K * sizeof(__nv_bfloat16);
  size_t mb_offset = (s_A_aligned + s_B_bytes + 7) & ~(size_t)7;
  size_t smem_size = mb_offset + NUM_STAGES * sizeof(mbarrier_t);

  cudaFuncSetAttribute(
      gemv_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size));

  gemv_ws_kernel<<<num_sms, BLOCK_SIZE, smem_size, stream>>>(
      static_cast<const __nv_bfloat16*>(A.data_ptr()),
      tma_B,
      static_cast<__nv_bfloat16*>(C.data_ptr()),
      static_cast<int>(N),
      static_cast<int>(K),
      num_tiles_k);
}
