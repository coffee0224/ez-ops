#include <tvm/ffi/tvm_ffi.h>
#include <cuda.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int WARP_SIZE = 32;
constexpr int COMPUTE_WARPS = 8;
constexpr int DMA_WARPS = 1;
constexpr int BLOCK_SIZE = (COMPUTE_WARPS + DMA_WARPS) * WARP_SIZE;  // 288
constexpr int TILE_K = 256;

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
// Persistent warp-specialized GEMV kernel
//   C[N] = A[K] @ B[N,K]
//
// Warp 0: DMA producer (loads A via cp.async.cg, B via TMA)
// Warps 1-8: Compute consumers (each warp handles one row)
//
// Single-buffered: each row group loads all K tiles via TMA, compute waits,
// then does full dot product. No double buffering — simpler, less shared memory.
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

  // Single buffer for B: COMPUTE_WARPS rows × K columns
  __nv_bfloat16* s_B = reinterpret_cast<__nv_bfloat16*>(smem_raw + s_A_aligned);

  size_t s_B_bytes = COMPUTE_WARPS * K * sizeof(__nv_bfloat16);
  size_t mb_offset = (s_A_aligned + s_B_bytes + 7) & ~(size_t)7;
  mbarrier_t* mb = reinterpret_cast<mbarrier_t*>(smem_raw + mb_offset);

  int num_blocks = gridDim.x;
  int total_row_groups = (N + COMPUTE_WARPS - 1) / COMPUTE_WARPS;
  uint32_t tile_bytes = COMPUTE_WARPS * TILE_K * sizeof(__nv_bfloat16);

  // ---- Init mbarrier ----
  if (threadIdx.x == 0) {
    mbarrier_init(&mb[0], 1);
  }
  __syncthreads();

  // ---- Phase 1: Load A into smem (cp.async.cg) ----
  if (is_dma) {
    int tile_16B = K / 8;
    for (int i = lane_id; i < tile_16B; i += WARP_SIZE) {
      uint32_t smem_addr = static_cast<uint32_t>(
          __cvta_generic_to_shared(s_A + i * 8));
      CP_ASYNC_CG(smem_addr, A + i * 8);
    }
    CP_ASYNC_COMMIT_GROUP();
    cp_async_wait_group_0();
    __syncwarp();
  }
  __syncthreads();

  // ---- Phase 2: Persistent loop ----
  uint32_t phase = 0;
  constexpr int PAIRS = TILE_K / (2 * WARP_SIZE);  // 4

  for (int rg = blockIdx.x; rg < total_row_groups; rg += num_blocks) {
    int row_base = rg * COMPUTE_WARPS;
    int my_row = row_base + comp_idx;

    // DMA: issue all TMA tiles for this row group
    if (threadIdx.x == 0 && rg < total_row_groups) {
      uint32_t total_bytes = num_tiles_k * tile_bytes;
      mbarrier_expect_tx(&mb[0], total_bytes);
      for (int kt = 0; kt < num_tiles_k; kt++) {
        cp_async_bulk_tensor_2d(
            &mb[0], &tma_B,
            s_B + kt * COMPUTE_WARPS * TILE_K,
            static_cast<int32_t>(kt * TILE_K),
            static_cast<int32_t>(row_base));
      }
    }

    // Compute: wait for TMA data
    if (!is_dma) {
      mbarrier_wait(&mb[0], phase);
      phase ^= 1;
    }

    // Compute: full dot product
    if (!is_dma && my_row < N) {
      float acc = 0.0f;
#pragma unroll
      for (int kt = 0; kt < (K / TILE_K); kt++) {
        const __nv_bfloat16* B_row_tile =
            s_B + kt * COMPUTE_WARPS * TILE_K + comp_idx * TILE_K;
        int k_start = kt * TILE_K;
#pragma unroll
        for (int i = 0; i < PAIRS; i++) {
          int idx = lane_id * 2 + i * WARP_SIZE * 2;
          __nv_bfloat162 b_val = *reinterpret_cast<const __nv_bfloat162*>(&B_row_tile[idx]);
          __nv_bfloat162 a_val = *reinterpret_cast<const __nv_bfloat162*>(&s_A[k_start + idx]);
          float2 av = __bfloat1622float2(a_val);
          float2 bv = __bfloat1622float2(b_val);
          acc += av.x * bv.x + av.y * bv.y;
        }
      }
      acc = warp_reduce_sum(acc);
      if (lane_id == 0) {
        C[my_row] = __float2bfloat16(acc);
      }
    }

    __syncthreads();
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
  size_t s_B_bytes = COMPUTE_WARPS * K * sizeof(__nv_bfloat16);
  size_t mb_offset = (s_A_aligned + s_B_bytes + 7) & ~(size_t)7;
  size_t smem_size = mb_offset + sizeof(mbarrier_t);

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
