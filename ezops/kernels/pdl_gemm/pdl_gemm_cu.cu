// Demo for Programmatic Dependent Launch (PDL).
//
// Computes two chained GEMMs:
//   y = x @ W1     (M,N) = (M,K) @ (K,N)
//   z = y @ W2     (M,P) = (M,N) @ (N,P)
//
// Two host entry points share the same templated GEMM kernel:
//   pdl_gemm_baseline_cu : launches both kernels normally (no PDL)
//   pdl_gemm_pdl_cu      : launches both with PDL launch attribute, and the
//                          kernel emits griddepcontrol.wait / launch_dependents
//                          PTX so that FC2's prolog overlaps with FC1's
//                          epilogue and FC1's grid-ending membar.

#include <tvm/ffi/tvm_ffi.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Tile config
// -----------------------------------------------------------------------------
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS_M = BM / TM;            // 16
constexpr int THREADS_N = BN / TN;            // 16
constexpr int NUM_THREADS = THREADS_M * THREADS_N;  // 256

// -----------------------------------------------------------------------------
// GEMM kernel
// -----------------------------------------------------------------------------
// C[M,N] = A[M,K] @ B[K,N]
//
// When USE_PDL is true:
//   * griddepcontrol.wait is inserted before the mainloop reads A/B, ensuring
//     we don't read stale outputs from the previous kernel in the stream.
//   * griddepcontrol.launch_dependents is inserted before the epilogue store,
//     so the next kernel's launch + prolog can overlap with this kernel's
//     epilogue + grid-ending membar.
template <bool USE_PDL>
__global__ void gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ float smem[];
    float (*sA)[BK] = reinterpret_cast<float (*)[BK]>(smem);
    float (*sB)[BN] = reinterpret_cast<float (*)[BN]>(smem + BM * BK);

    const int tid = threadIdx.x;
    const int thread_m = tid / THREADS_N;
    const int thread_n = tid % THREADS_N;

    const int bm_start = blockIdx.y * BM;
    const int bn_start = blockIdx.x * BN;

    float acc[TM][TN] = {};

    // PDL: block mainloop until previous kernel's output is visible in global mem.
    if constexpr (USE_PDL) {
        asm volatile("griddepcontrol.wait;");
    }

    // Mainloop over K tiles
    const int num_k_tiles = (K + BK - 1) / BK;
    bool launched_dependents = false;
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        // Cooperative load: BM*BK + BK*BN = 2048 floats, 8 per thread.
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx_a = tid + i * NUM_THREADS;  // 0..1023
            int sa_row = idx_a / BK;
            int sa_col = idx_a % BK;
            int ga_row = bm_start + sa_row;
            int ga_col = k_tile * BK + sa_col;
            sA[sa_row][sa_col] =
                (ga_row < M && ga_col < K) ? A[ga_row * K + ga_col] : 0.0f;

            int sb_row = idx_a / BN;
            int sb_col = idx_a % BN;
            int gb_row = k_tile * BK + sb_row;
            int gb_col = bn_start + sb_col;
            sB[sb_row][sb_col] =
                (gb_row < K && gb_col < N) ? B[gb_row * N + gb_col] : 0.0f;
        }
        __syncthreads();

        // PDL: launch FC2 once we're roughly halfway through the K loop, so
        // FC2's prolog (smem alloc, constant load) can overlap with the rest
        // of our mainloop + epilogue + grid-ending membar. Earlier placement
        // gives more overlap window but risks FC2's prolog stealing SM
        // resources from our mainloop; midway is a safe default.
        if constexpr (USE_PDL) {
            if (!launched_dependents && k_tile + 1 >= num_k_tiles / 2) {
                asm volatile("griddepcontrol.launch_dependents;");
                launched_dependents = true;
            }
        }

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a_vals[TM];
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                a_vals[m] = sA[thread_m * TM + m][kk];
            }
            float b_vals[TN];
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                b_vals[n] = sB[kk][thread_n * TN + n];
            }
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    acc[m][n] += a_vals[m] * b_vals[n];
                }
            }
        }
        __syncthreads();
    }

    // PDL: if we never hit the midway point (very short K), fall back to
    // launching at the end so the next kernel still gets queued before our
    // grid-ending membar.
    if constexpr (USE_PDL) {
        if (!launched_dependents) {
            asm volatile("griddepcontrol.launch_dependents;");
        }
    }

    // Epilogue: store C[bm_start:bm_start+BM, bn_start:bn_start+BN]
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int gm_row = bm_start + thread_m * TM + m;
            int gm_col = bn_start + thread_n * TN + n;
            if (gm_row < M && gm_col < N) {
                C[gm_row * N + gm_col] = acc[m][n];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
static inline cudaStream_t get_stream(tvm::ffi::TensorView x) {
    DLDevice dev = x.device();
    return static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));
}

static inline dim3 grid_for(int M, int out_cols) {
    return dim3((out_cols + BN - 1) / BN, (M + BM - 1) / BM);
}

// -----------------------------------------------------------------------------
// Baseline: no PDL. Two serial kernel launches on the same stream.
// -----------------------------------------------------------------------------
void pdl_gemm_baseline_cu(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView W1,
    tvm::ffi::TensorView W2,
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView z)
{
    int M = static_cast<int>(x.size(0));
    int K = static_cast<int>(x.size(1));
    int N = static_cast<int>(W1.size(1));
    int P = static_cast<int>(W2.size(1));

    const float* x_ptr  = static_cast<const float*>(x.data_ptr());
    const float* W1_ptr = static_cast<const float*>(W1.data_ptr());
    const float* W2_ptr = static_cast<const float*>(W2.data_ptr());
    float* y_ptr = static_cast<float*>(y.data_ptr());
    float* z_ptr = static_cast<float*>(z.data_ptr());

    cudaStream_t stream = get_stream(x);
    dim3 block(NUM_THREADS);
    size_t smem = (BM * BK + BK * BN) * sizeof(float);

    // FC1: y = x @ W1
    gemm_kernel<false><<<grid_for(M, N), block, smem, stream>>>(
        x_ptr, W1_ptr, y_ptr, M, N, K);

    // FC2: z = y @ W2
    gemm_kernel<false><<<grid_for(M, P), block, smem, stream>>>(
        y_ptr, W2_ptr, z_ptr, M, P, N);
}

// -----------------------------------------------------------------------------
// PDL: launch both kernels with ProgrammaticStreamSerialization attribute,
// kernel emits griddepcontrol PTX to overlap FC2's prolog with FC1's epilogue.
// -----------------------------------------------------------------------------
void pdl_gemm_pdl_cu(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView W1,
    tvm::ffi::TensorView W2,
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView z)
{
    int M = static_cast<int>(x.size(0));
    int K = static_cast<int>(x.size(1));
    int N = static_cast<int>(W1.size(1));
    int P = static_cast<int>(W2.size(1));

    const float* x_ptr  = static_cast<const float*>(x.data_ptr());
    const float* W1_ptr = static_cast<const float*>(W1.data_ptr());
    const float* W2_ptr = static_cast<const float*>(W2.data_ptr());
    float* y_ptr = static_cast<float*>(y.data_ptr());
    float* z_ptr = static_cast<float*>(z.data_ptr());

    cudaStream_t stream = get_stream(x);
    dim3 block(NUM_THREADS);
    size_t smem = (BM * BK + BK * BN) * sizeof(float);

    // PDL launch attribute: tells the runtime this kernel may be overlapped
    // with the previous one in the same stream.
    cudaLaunchAttribute pdl_attr;
    pdl_attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    pdl_attr.val.programmaticStreamSerializationAllowed = 1;

    auto make_config = [&](dim3 grid) {
        cudaLaunchConfig_t c{};
        c.gridDim = grid;
        c.blockDim = block;
        c.dynamicSmemBytes = smem;
        c.stream = stream;
        c.attrs = &pdl_attr;
        c.numAttrs = 1;
        return c;
    };

    cudaLaunchConfig_t cfg1 = make_config(grid_for(M, N));
    cudaLaunchConfig_t cfg2 = make_config(grid_for(M, P));

    // FC1: y = x @ W1, with PDL.
    cudaLaunchKernelEx(&cfg1, gemm_kernel<true>,
        x_ptr, W1_ptr, y_ptr, M, N, K);

    // FC2: z = y @ W2, with PDL. The kernel's griddepcontrol.wait will block
    // until FC1's grid-ending membar has committed y to global memory.
    cudaLaunchKernelEx(&cfg2, gemm_kernel<true>,
        y_ptr, W2_ptr, z_ptr, M, P, N);
}
