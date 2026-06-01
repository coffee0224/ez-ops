# GPU Kernel Optimization Strategies

This document provides detailed optimization strategies organized by bottleneck type. Each strategy includes the problem it solves, how to implement it, and expected gains.

## Table of Contents

1. [Memory-Bound Optimizations](#memory-bound-optimizations)
2. [Compute-Bound Optimizations](#compute-bound-optimizations)
3. [Occupancy-Bound Optimizations](#occupancy-bound-optimizations)
4. [Pipeline/Latency Optimizations](#pipelinelatency-optimizations)
5. [Algorithmic Optimizations](#algorithmic-optimizations)
6. [Hardware-Specific (SM90+)](#hardware-specific-sm90)

---

## Memory-Bound Optimizations

### 1. Vectorized Memory Access

**When to use**: NCU shows low DRAM throughput despite many load instructions. Loads are narrow (1-byte or 4-byte per transaction).

**What it does**: Use wider load types (uint4 = 16 bytes, float4 = 16 bytes) to increase bytes per memory transaction.

**CUDA implementation**:
```cuda
// Before: scalar loads
float val = B[row * K + col];

// After: vectorized loads
float4 vals = reinterpret_cast<float4*>(B)[idx]; // loads 4 floats at once
```

**Expected gain**: 1.5-3x for bandwidth-limited kernels.

### 2. Memory Coalescing

**When to use**: NCU shows high "sectors per request" ratio (>2). Threads in a warp access non-contiguous memory.

**What it does**: Restructure so that consecutive threads access consecutive memory addresses.

**Implementation**: Reorder data layout (e.g., column-major to row-major) or restructure thread-to-data mapping.

**Expected gain**: 1.5-3x for uncoalesced patterns.

### 3. Shared Memory Tiling

**When to use**: Same data is loaded from global memory multiple times. NCU shows high DRAM read bytes relative to working set.

**What it does**: Load data into shared memory once, reuse across threads. Use `__syncthreads()` to synchronize.

```cuda
__shared__ float tile[TILE_SIZE][TILE_SIZE];
tile[ty][tx] = global_data[row * width + col];
__syncthreads();
// Multiple threads read from tile without going to global memory
```

**Expected gain**: 1.2-3x depending on data reuse factor.

### 4. Shared Memory Bank Conflict Avoidance

**When to use**: NCU shows high shared memory bank conflicts (check shared load/store throughput).

**What it does**: Pad shared memory arrays to avoid multiple threads accessing the same bank in the same cycle.

```cuda
// Before: 32 banks, width=32 may cause conflicts
__shared__ float tile[32][32];

// After: pad to 33 to break bank conflicts
__shared__ float tile[32][33];
```

**Expected gain**: 1.1-1.3x.

### 5. Cache-Aware Access Patterns

**When to use**: NCU shows low L1/L2 cache hit rates.

**What it does**: Structure accesses to maximize cache line reuse before eviction.

**Techniques**:
- Access data in cache-line-sized blocks (128 bytes)
- Process data sequentially rather than randomly
- Use `__ldg()` for read-only data (goes through texture cache)
- Use `__ldcs()`, `__ldcg()`, `__ldca()` cache hints

**Expected gain**: 1.2-2x.

---

## Compute-Bound Optimizations

### 1. Tensor Core Utilization (HMMA/WMMA)

**When to use**: Kernel performs matrix multiply or dot products. NCU shows high FMA pipe utilization but no tensor core usage.

**What it does**: Use WMMA (CUDA built-in) or HMMA (PTX inline assembly) to leverage tensor cores for 2-8x throughput on matrix operations.

```cuda
#include <mma.h>
using namespace nvcuda::wmma;
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;
load_matrix_sync(a_frag, a_ptr, 16);
load_matrix_sync(b_frag, b_ptr, 16);
fill_fragment(c_frag, 0.0f);
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(c_ptr, c_frag, 16, mem_row_major);
```

**Expected gain**: 2-8x for matmul/dot-product heavy kernels.

### 2. Operation Fusion

**When to use**: Multiple kernel launches for chained operations. NCU shows low compute-to-memory ratio.

**What it does**: Combine multiple operations into a single kernel to eliminate intermediate memory writes/reads.

**Expected gain**: 1.3-3x (eliminates memory bandwidth for intermediates).

### 3. Instruction-Level Parallelism (ILP)

**When to use**: Low IPC (<1.0). Long dependency chains between instructions.

**What it does**: Restructure computation so independent operations can issue back-to-back without stalls.

```cuda
// Before: sequential with dependencies
for (int i = 0; i < N; i++) {
    sum += data[i] * data[i];
}

// After: unrolled with independent accumulators
float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
for (int i = 0; i < N; i += 4) {
    sum0 += data[i] * data[i];
    sum1 += data[i+1] * data[i+1];
    sum2 += data[i+2] * data[i+2];
    sum3 += data[i+3] * data[i+3];
}
float sum = sum0 + sum1 + sum2 + sum3;
```

**Expected gain**: 1.2-2x.

---

## Occupancy-Bound Optimizations

### 1. Register Pressure Reduction

**When to use**: NCU shows low occupancy due to high register usage (check `Registers/Thread` and occupancy limiter).

**What it does**: Reduce per-thread register count to allow more warps per SM.

**Techniques**:
- Use `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` to guide compiler
- Reuse variables instead of declaring many temporaries
- Split complex kernels into simpler passes
- Check compilation output with `-Xptxas -v` to see register allocation

**Expected gain**: 1.2-2x from improved occupancy.

### 2. Block/Grid Configuration Tuning

**When to use**: NCU shows low `Waves/SM` (<2) or low grid size.

**What it does**: Adjust block and grid dimensions to maximize SM utilization.

**Guidelines**:
- Block size should be multiple of warp size (32)
- Aim for at least 8-16 blocks per SM
- For small workloads, consider splitting work across more blocks (e.g., split-K for GEMV)
- Use 256 or 512 threads per block as starting point

**Expected gain**: 1.2-1.5x.

### 3. Dynamic Shared Memory

**When to use**: Static shared memory allocation limits occupancy.

**What it does**: Allocate only the shared memory needed per launch configuration.

```cuda
// Dynamic shared memory
extern __shared__ float dynamic_tile[];
// Launch with: kernel<<<grid, block, shared_mem_bytes>>>(...)
```

**Expected gain**: 1.1-1.3x if shared memory was the occupancy limiter.

---

## Pipeline/Latency Optimizations

### 1. Warp-Specialization

**When to use**: Kernel has distinct producer (load) and consumer (compute) phases. NCU shows high stall on wait/barrier.

**What it does**: Dedicate warps to specific roles — some warps only load data, others only compute. Synchronize via barriers or `__pipeline_*` primitives.

**Expected gain**: 1.3-2x from overlapping memory and compute.

### 2. Double Buffering

**When to use**: Kernel processes data in tiles with clear phase boundaries.

**What it does**: While computing on tile N, load tile N+1 into a separate buffer. Eliminates idle time between load and compute phases.

```cuda
__shared__ float buffer[2][TILE_SIZE];
int current = 0;
// Load first tile
load_tile(buffer[current], ...);
__syncthreads();
for (int tile = 0; tile < num_tiles; tile++) {
    int next = 1 - current;
    if (tile + 1 < num_tiles)
        load_tile(buffer[next], ...);  // async or in background
    compute(buffer[current], ...);
    __syncthreads();
    current = next;
}
```

**Expected gain**: 1.2-1.5x.

### 3. Async Memory Copies (cp.async)

**When to use**: SM80+ (Ampere and later). Kernel loads data from global to shared memory.

**What it does**: Use `cp.async` to overlap global-to-shared transfers with computation.

```cuda
// Async copy from global to shared
asm volatile("cp.async.cg.shared.global [%0], [%1], %2;" ::
    "r"(smem_ptr), "l"(gmem_ptr), "n"(16));
// Wait for N pending async copies
asm volatile("cp.async.commit_group;");
asm volatile("cp.async.wait_group %0;" :: "n"(0));
```

**Expected gain**: 1.2-1.5x for memory-heavy kernels.

---

## Algorithmic Optimizations

### 1. Work Reduction

**When to use**: Kernel performs redundant computation that can be eliminated.

**Examples**:
- Split-K reduction: each thread block computes partial result, then reduce
- Early exit: skip computation when result is known (e.g., zero weights)
- Approximate computation where exact result isn't needed

**Expected gain**: 1.2-3x depending on redundancy.

### 2. Parallel Reduction Patterns

**When to use**: Kernel has reduction steps with poor parallelism.

**What it does**: Use efficient parallel reduction patterns (butterfly, warp shuffle, etc.).

```cuda
// Warp-level reduction using shuffle
float val = thread_data;
for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
```

**Expected gain**: 1.2-2x for reduction-heavy kernels.

---

## Hardware-Specific (SM90+)

### 1. TMA (Tensor Memory Accelerator)

**When to use**: SM90+ (Hopper, Blackwell). Kernel loads multi-dimensional data from global memory.

**What it does**: Use TMA descriptors to describe data layout and let hardware handle addressing, broadcasting, and boundary checking.

```cuda
// TMA load (simplified)
CUtensorMap tensor_map;
cuTensorMapEncodeTiled(&tensor_map, ...);
// In kernel:
CP_ASYNC_BULK_KERNEL_3D(smem_ptr, &tensor_map, ...);
```

**Expected gain**: 1.2-2x from reduced address calculation overhead and hardware-optimized transfers.

### 2. Cluster Groups

**When to use**: SM90+. Kernel needs cooperation between thread blocks.

**What it does**: Use thread block clusters (2-8 blocks that can synchronize and share distributed shared memory).

**Expected gain**: 1.1-1.3x from cross-block cooperation.

### 3. Warp-Specialized with mbarrier

**When to use**: SM90+. Producer-consumer pattern between warps.

**What it does**: Use `mbarrier` for lightweight warp-to-warp synchronization, combined with TMA or cp.async.

**Expected gain**: 1.3-2x from efficient pipeline synchronization.

### 4. Dynamic Shared Memory with Distributed Shared Memory

**When to use**: SM90+. Working set exceeds per-block shared memory.

**What it does**: Use distributed shared memory across thread blocks in a cluster.

**Expected gain**: Enables previously impossible optimizations for large working sets.
