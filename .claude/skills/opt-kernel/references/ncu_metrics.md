# Key NCU Metrics Guide

This reference explains the most important NCU metrics for diagnosing GPU kernel performance issues. Organized by category.

## Timing Metrics

| Metric | What It Means | What to Look For |
|--------|---------------|------------------|
| `gpu__time_duration.sum` | Total kernel execution time in nanoseconds | The bottom-line number to optimize |
| `smsp__cycles_active.avg` | Average cycles SMs are active | Compare against theoretical max cycles |

## SOL (Speed-of-Light) Throughput

These metrics show how close the kernel gets to peak hardware throughput. They're the most important high-level indicators.

| Metric | What It Means | Target |
|--------|---------------|--------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM compute pipeline utilization | >80% = compute bound |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | L2-to-DRAM throughput | >60% = memory bound |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM bandwidth utilization | >60% = memory bound |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1/texture cache throughput | High = shared mem or cache heavy |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | Overall memory throughput | Combined indicator |

**Interpretation**:
- If SM throughput is high but memory is low: **compute-bound** (good! but maybe use tensor cores)
- If memory throughput is high but SM is low: **memory-bound** (optimize access patterns)
- If both are low: **occupancy or latency bound** (increase parallelism or reduce sync)

## Occupancy Metrics

| Metric | What It Means | What to Look For |
|--------|---------------|------------------|
| `launch__grid_size` | Total number of blocks launched | Should be >> number of SMs |
| `launch__block_size` | Threads per block | Should be multiple of 32 (warp size) |
| `launch__waves_per_multiprocessor` | How many times each SM processes the kernel | >2 is good, <1 means underutilization |
| `launch__registers_per_thread` | Register allocation per thread | High values limit occupancy |
| `launch__shared_memory_per_block` | Shared memory per block | High values limit occupancy |
| `sm__warps_active.avg.pct_of_peak` | Theoretical occupancy | Compare with achieved |
| `smsp__warps_active.avg.pct_of_peak` | Achieved occupancy | Should be close to theoretical |

**Occupancy rules of thumb**:
- <30% achieved occupancy: likely a problem
- Registers/thread >64: will significantly limit occupancy
- Block size <128: consider increasing

## Stall Reason Metrics

These are the most diagnostic metrics. They tell you *why* warps are stalling. Each is expressed as a percentage of issue slots.

| Metric | What Causes It | Fix |
|--------|----------------|-----|
| `stalled_long_scoreboard` | Waiting for global memory load to return | Reduce load-to-use distance, prefetch, use more independent work |
| `stalled_short_scoreboard` | Waiting for shared memory or other short-latency ops | Reduce shared memory bank conflicts, reduce shared mem pressure |
| `stalled_wait` | Waiting on a barrier or sync | Reduce synchronization frequency, use warp-level primitives |
| `stalled_math_pipe_throttle` | Compute pipelines are full | Good sign if memory isn't the bottleneck, consider tensor cores |
| `stalled_mio_throttle` | Memory I/O pipeline is backed up | Reduce memory instruction rate, batch operations |
| `stalled_lg_throttle` | Load/store unit is overloaded | Reduce per-warp memory instructions, vectorize |
| `stalled_not_selected` | Warp is eligible but scheduler chose another warp | Increase occupancy to give scheduler more options |
| `stalled_barrier` | Waiting at `__syncthreads()` | Reduce sync frequency, use warp-level communication |
| `stalled_branch_resolving` | Waiting for branch resolution | Reduce divergent branches |
| `stalled_no_instruction` | No instruction available to issue | Increase ILP, prefetch data |
| `stalled_dispatch_stall` | Waiting for next dispatch from GPC | Usually a launch configuration issue |
| `stalled_membar` | Waiting for memory barrier | Reduce memory barrier usage |

**Priority**: Focus on the top 2-3 stall reasons. A stall at 20%+ is significant. The sum of all stall percentages plus the active percentage should roughly equal 100%.

## Memory Metrics

| Metric | What It Means | What to Look For |
|--------|---------------|------------------|
| `dram__bytes_read.sum` | Total bytes read from DRAM | Compare with working set size |
| `dram__bytes_write.sum` | Total bytes written to DRAM | Should be close to output size |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` | Shared memory load wavefronts | High + bank conflicts = bad |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum` | Shared memory store wavefronts | |
| `lts__sectors_op_read.sum` | L2 read sectors | High sectors per request = poor coalescing |
| `lts__sectors_op_write.sum` | L2 write sectors | |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | L1 sectors for global loads | High = poor coalescing or over-fetching |
| `smsp__inst_executed.sum.op_global_load` | Global load instructions | Compare with useful work instructions |
| `smsp__inst_executed.sum.op_global_store` | Global store instructions | |
| `smsp__inst_executed.sum.op_shared_load` | Shared memory load instructions | |
| `smsp__inst_executed.sum.op_shared_store` | Shared memory store instructions | |

**Coalescing check**: If `sectors per global load` > 4, memory accesses are poorly coalesced. Each warp should ideally access 1-4 sectors per load instruction (32 threads * 4 bytes = 128 bytes = 2 sectors on most GPUs).

## Compute Metrics

| Metric | What It Means | What to Look For |
|--------|---------------|------------------|
| `smsp__issue_active.avg.inst_per_cycle` | Instructions issued per cycle (IPC) | Peak is ~4 for most architectures |
| `smsp__inst_executed.sum.op_ffma` | FMA instructions | Core compute instructions |
| `smsp__inst_executed.sum.alu` | ALU instructions | |
| `smsp__pipeline_tensor_op_hmma_active.avg.pct_of_peak` | Tensor core utilization | Should be high for matmul workloads |

## Quick Diagnosis Cheat Sheet

```
1. Check SM throughput and memory throughput
   - SM high, mem low → Compute bound → Use tensor cores or fuse ops
   - Mem high, SM low → Memory bound → Vectorize, coalesce, cache
   - Both low → Occupancy bound → Increase parallelism

2. Check top stall reason
   - Long scoreboard → Memory latency → Prefetch, independent work
   - LG throttle → Memory bandwidth → Vectorize, reduce traffic
   - Math pipe → Compute saturation → Tensor cores
   - Not selected → Low occupancy → More blocks/threads
   - Barrier → Over-synchronization → Reduce sync, warp-specialize

3. Check occupancy
   - Waves/SM < 2 → Increase grid or block size
   - Achieved < 50% of theoretical → Register or shared mem pressure

4. Check memory coalescing
   - Sectors per load > 4 → Fix access pattern
   - DRAM bytes >> working set → Data is being re-fetched, use shared memory
```
