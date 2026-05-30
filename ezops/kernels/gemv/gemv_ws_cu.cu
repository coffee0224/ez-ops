#include <tvm/ffi/tvm_ffi.h>
#include <cuda_bf16.h>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
constexpr int DMA_WARPS = 1;
constexpr int COMPUTE_WARPS = NUM_WARPS - DMA_WARPS;
constexpr int TILE_K = 256;
constexpr int NUM_STAGES = 2;
constexpr int CP16B_ELEMS = 8;  // 16 bytes = 8 x bfloat16

// ---------------------------------------------------------------------------
// PTX helpers: cp.async (SM80+) — matches SM120 blog pattern
// ---------------------------------------------------------------------------

#define CP_ASYNC_CG(dst_smem_32b, src_global_ptr)                                              \
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst_smem_32b),     \
               "l"(src_global_ptr))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  if constexpr (N == 0)
    asm volatile("cp.async.wait_group 0;\n" ::);
  else if constexpr (N == 1)
    asm volatile("cp.async.wait_group 1;\n" ::);
  else if constexpr (N == 2)
    asm volatile("cp.async.wait_group 2;\n" ::);
}

// ---------------------------------------------------------------------------
// PTX helpers: mbarrier (SM90+/SM120)
// ---------------------------------------------------------------------------

typedef uint64_t mbarrier_t;

__device__ __forceinline__ void mbarrier_init(mbarrier_t* mb, uint32_t count) {
  asm volatile(
      "mbarrier.init.shared.b64 [%0], %1;\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mb))), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive(mbarrier_t* mb) {
  asm volatile(
      "mbarrier.arrive.shared.b64 _, [%0];\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mb))));
}

__device__ __forceinline__ void mbarrier_wait(mbarrier_t* mb, uint32_t phase) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mb));
  uint32_t ticks = 0x989680;  // ~10M cycle timeout
  asm volatile(
      "{\n\t"
      ".reg .pred p; \n\t"
      "LAB_WAIT: \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2; \n\t"
      "@p bra DONE; \n\t"
      "bra LAB_WAIT; \n\t"
      "DONE: \n\t"
      "}\n"
      :
      : "r"(smem_addr), "r"(phase), "r"(ticks)
      : "memory");
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
// DMA helper: async-load one K-tile of B rows via cp.async.cg
// All 32 DMA threads cooperate. Issues cp.async per 16B chunk.
// Caller must commit_group + wait_group to ensure completion.
// ---------------------------------------------------------------------------

__device__ void async_load_tile(
    __nv_bfloat16* smem_buf,
    const __nv_bfloat16* __restrict__ B,
    int row_base,
    int k_tile,
    int N,
    int K) {
  int k_start = k_tile * TILE_K;
  int k_end = min(k_start + TILE_K, K);
  int tile_16B = (k_end - k_start) / CP16B_ELEMS;

  int lane_id = threadIdx.x % WARP_SIZE;

  for (int row = 0; row < COMPUTE_WARPS; row++) {
    int gmem_row = row_base + row;
    if (gmem_row >= N) break;

    const __nv_bfloat16* gmem_ptr = B + gmem_row * K + k_start;
    __nv_bfloat16* smem_ptr = smem_buf + row * TILE_K;

    for (int i = lane_id; i < tile_16B; i += WARP_SIZE) {
      uint32_t smem_addr =
          static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr + i * CP16B_ELEMS));
      CP_ASYNC_CG(smem_addr, gmem_ptr + i * CP16B_ELEMS);
    }
    // Tail: remaining (< 8) elements via synchronous load
    for (int k = tile_16B * CP16B_ELEMS + lane_id; k < (k_end - k_start); k += WARP_SIZE) {
      smem_ptr[k] = __ldg(gmem_ptr + k);
    }
  }
}

// ---------------------------------------------------------------------------
// Warp-specialized GEMV kernel
//   C[N] = A[K] @ B[N,K]   (bfloat16 I/O, fp32 accumulation)
//
// Warp 0:    DMA  (loads A, then async-streams B tiles into double buffer)
// Warps 1-7: Compute (dot product with double-buffered B tiles)
//
// Synchronization:
//   DMA warp: cp.async.cg loads → commit_group → arrive on mbarrier
//   Compute warps: mbarrier_wait (blocks until DMA arrival + cp.async done)
//   Per-buffer phase tracking for correct double-buffered parity
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 2) gemv_ws_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int N,
    int K) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  bool is_dma = (warp_id == 0);
  int comp_idx = warp_id - 1;

  int num_tiles = (K + TILE_K - 1) / TILE_K;

  // ---- Dynamic shared memory layout ----
  extern __shared__ char smem_raw[];
  float* s_A = reinterpret_cast<float*>(smem_raw);
  size_t s_A_bytes = (size_t)K * sizeof(float);

  __nv_bfloat16* s_B =
      reinterpret_cast<__nv_bfloat16*>(smem_raw + s_A_bytes);
  size_t s_B_buf_bytes =
      (size_t)COMPUTE_WARPS * TILE_K * sizeof(__nv_bfloat16);

  __nv_bfloat16* s_B_buf[NUM_STAGES] = {
      s_B,
      s_B + COMPUTE_WARPS * TILE_K,
  };

  mbarrier_t* mb = reinterpret_cast<mbarrier_t*>(
      smem_raw + s_A_bytes + NUM_STAGES * s_B_buf_bytes);

  int row_base = blockIdx.x * COMPUTE_WARPS;
  int my_row = row_base + comp_idx;

  // ---- Initialize mbarriers (1 arrival expected: the DMA warp) ----
  if (threadIdx.x < NUM_STAGES) {
    mbarrier_init(&mb[threadIdx.x], 1);
  }
  __syncthreads();

  // ---- Phase 1: DMA warp loads vector A into s_A (fp32) ----
  if (is_dma) {
    for (int i = lane_id; i < K; i += WARP_SIZE) {
      s_A[i] = __bfloat162float(__ldg(A + i));
    }
  }
  __syncthreads();

  // ---- Phase 2: DMA warp async-loads first B tile into buf0 ----
  if (is_dma && num_tiles > 0) {
    async_load_tile(s_B_buf[0], B, row_base, 0, N, K);
    CP_ASYNC_COMMIT_GROUP();
    cp_async_wait_group<0>();
    __syncwarp();
    if (lane_id == 0) mbarrier_arrive(&mb[0]);
  }

  // ---- Phase 3: Main pipeline loop ----
  float acc = 0.0f;
  uint32_t wait_phase[NUM_STAGES] = {0, 0};

  for (int k_tile = 0; k_tile < num_tiles; k_tile++) {
    int cur_buf = k_tile & 1;
    int nxt_buf = cur_buf ^ 1;

    if (!is_dma && my_row < N) {
      // --- Compute warps: wait for cur_buf, then dot product ---
      mbarrier_wait(&mb[cur_buf], wait_phase[cur_buf]);

      const __nv_bfloat16* B_row = s_B_buf[cur_buf] + comp_idx * TILE_K;
      int k_start = k_tile * TILE_K;
      int k_end = min(k_start + TILE_K, K);
      int tile_len = k_end - k_start;

#pragma unroll 4
      for (int k = lane_id; k < tile_len; k += WARP_SIZE) {
        acc += s_A[k_start + k] * __bfloat162float(B_row[k]);
      }
    }

    if (is_dma && (k_tile + 1 < num_tiles)) {
      // --- DMA warp: async-load next tile into nxt_buf ---
      async_load_tile(s_B_buf[nxt_buf], B, row_base, k_tile + 1, N, K);
      CP_ASYNC_COMMIT_GROUP();
      cp_async_wait_group<0>();
      __syncwarp();
      if (lane_id == 0) mbarrier_arrive(&mb[nxt_buf]);
    }

    wait_phase[cur_buf] ^= 1;
  }

  // ---- Phase 4: Warp reduce and write output ----
  if (!is_dma && my_row < N) {
    acc = warp_reduce_sum(acc);
    if (lane_id == 0) {
      C[my_row] = __float2bfloat16(acc);
    }
  }
}

// ---------------------------------------------------------------------------
// Host entry point (called from Python via tvm_ffi)
// ---------------------------------------------------------------------------

void gemv_ws_cu(
    tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
  int64_t K = A.size(0);
  int64_t N = B.size(0);

  int num_blocks = (N + COMPUTE_WARPS - 1) / COMPUTE_WARPS;

  size_t s_A_bytes = (size_t)K * sizeof(float);
  size_t s_B_bytes =
      (size_t)NUM_STAGES * COMPUTE_WARPS * TILE_K * sizeof(__nv_bfloat16);
  size_t mb_bytes = NUM_STAGES * sizeof(mbarrier_t);
  size_t smem_size = s_A_bytes + s_B_bytes + mb_bytes;

  DLDevice dev = A.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  cudaFuncSetAttribute(
      gemv_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  gemv_ws_kernel<<<num_blocks, BLOCK_SIZE, smem_size, stream>>>(
      static_cast<const __nv_bfloat16*>(A.data_ptr()),
      static_cast<const __nv_bfloat16*>(B.data_ptr()),
      static_cast<__nv_bfloat16*>(C.data_ptr()),
      static_cast<int>(N),
      static_cast<int>(K));
}
