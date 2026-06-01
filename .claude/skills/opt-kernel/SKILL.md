---
name: opt-kernel
description: >
  GPU kernel optimization workflow for the ez-ops framework. Guides Claude through a rigorous
  profile-driven optimization loop: benchmark baseline, NCU profile, diagnose bottlenecks,
  implement optimizations, verify correctness, and iterate. Use this skill whenever the user
  asks to optimize a GPU kernel, speed up a CUDA/Triton/TileLang kernel, improve kernel
  performance, profile a kernel with NCU, or anything related to GPU kernel tuning and
  performance engineering — even if they just say "make this kernel faster" or "optimize gemv"
  or "ncu profile this". Also trigger for Chinese phrases like "优化kernel", "加速一下",
  "性能调优", "profiling".
---

# GPU Kernel Optimization for ez-ops

## Golden Rule

**Profile -> Diagnose -> Plan -> Implement -> Verify. Never guess.**

Every optimization decision must be backed by data from benchmarks and NCU reports. If you're tempted to "just try" an optimization without profiling first, stop and profile instead.

## Input Parameters

When the user invokes this skill, extract or ask for these parameters:

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `kernel` | Yes | - | The op name to optimize (e.g. `gemv`, `attn_decode`, `vector_add`) |
| `direction` | No | Highest-impact first | Optimization focus area (e.g. "memory bandwidth", "occupancy", "compute throughput") |
| `dsl` | No | Any available | Allowed implementation DSL: `cuda`, `triton`, `tilelang`, or combinations |
| `baseline` | No | `ref` (the op's `_ref_forward`) | Which registered backend to use as baseline |

If the user doesn't specify all parameters, infer from context or ask. A minimal invocation looks like: "optimize gemv kernel" — this means optimize the `gemv` op, starting from the reference implementation, using any DSL.

## Understanding the ez-ops Framework

Before optimizing, understand how the project is organized. Read this section carefully.

**Operations (ops/)**: Each op (`GemvOp`, `AttnDecodeOp`, etc.) is a subclass of `Op` in `ezops/ops/`. The key methods are:
- `_ref_forward()` — PyTorch reference implementation (the default baseline)
- `forward()` — dispatches to the registered backend kernel
- `gen_data()` — generates test input tensors
- `check(actual, expected)` — verifies correctness with configurable tolerances

**Kernels (kernels/)**: Backend implementations live in `ezops/kernels/<op_name>/`. Each file contains a class decorated with `@register_kernel(op_name, backend_name)`. The registry maps `(op_name, backend_name)` to kernel classes.

**CUDA kernels** use `.cu` source files loaded via `tvm_ffi.cpp.load_inline()`. The ABI boundary uses `tvm::ffi::TensorView`.

**TileLang kernels** use `@tilelang.jit(out_idx=[2])` with `T.prim_func` definitions, built once in `__init__` via `_make_kernel()`.

**Benchmarks**: `benchmarks/bench_<op>.py` runs all registered backends, measures latency, computes SOL (Speed-of-Light) scores, and prints performance tables.

**Profiling**: `scripts/ncu_profile.py` runs `ncu --set full` on a specific backend, saving `.ncu-rep` files to `.profiles/`.

**Registration pattern** — to add a new backend:
```python
from ezops.registry import register_kernel
from ezops.kernels.base_kernel import BaseKernel

@register_kernel("op_name", "my_new_backend")
class MyNewKernel(BaseKernel):
    def __init__(self, *args, **kwargs):
        # Parse params, build/compile kernel
        ...
    def __call__(self, *args):
        # Run kernel
        ...
```
Then import the class in `ezops/kernels/<op_name>/__init__.py` so the decorator fires at import time.

## Optimization Workflow

### Phase 0: Setup

1. **Create a git branch** for this optimization session:
   ```bash
   git checkout -b opt/<kernel_name>_<timestamp>
   ```

2. **Record the goal**: what metric are we optimizing (latency, throughput, SOL score)?

3. **Run the first NCU analysis with `--kernel`** — this auto-creates the optimization log at `.profiles/<kernel>_opt_log.md`:
   ```bash
   python <skill-path>/scripts/analyze_ncu.py \
     --report .profiles/<report>.ncu-rep \
     --tag baseline \
     --kernel-name <kernel_name_substring> \
     --kernel <kernel>
   ```
   The `--kernel` flag triggers log creation and appending. Every subsequent analyze/compare run with `--kernel` appends an iteration record.

### Phase 1: Establish Baseline

The main agent does this directly (not via sub-agent).

**Step 1a: Run benchmarks**
```bash
cd /home/coffee/ez-ops
python benchmarks/bench_<kernel_name>.py
```
Record the baseline backend's latency and SOL score from the output.

**Step 1b: Profile with NCU**
```bash
python scripts/ncu_profile.py <kernel_name> -k <baseline_backend> -p <params>
```
This generates a `.ncu-rep` file in `.profiles/`. Note the filename.

**Step 1c: Analyze the NCU report**

Read the `.ncu-rep` file using the bundled `scripts/analyze_ncu.py` helper:
```bash
# List kernels in the report to find the target kernel
python <skill-path>/scripts/analyze_ncu.py --report .profiles/<report>.ncu-rep --tag baseline --list-kernels

# Analyze a specific kernel by name (recommended)
python <skill-path>/scripts/analyze_ncu.py \
  --report .profiles/<report>.ncu-rep \
  --tag baseline \
  --kernel-name <kernel_name_substring>

# Without --kernel-name, auto-selects the longest-duration kernel (may pick wrong one if report contains helper kernels)
python <skill-path>/scripts/analyze_ncu.py --report .profiles/<report>.ncu-rep --tag baseline
```

NCU reports often contain multiple kernels (e.g. PyTorch data generation kernels). **Always use `--list-kernels` first** to see what's in the report, then use `--kernel-name` to select the target. The auto-select mode picks the longest single-kernel duration, which can be misleading when helper kernels run sequentially.

The analysis extracts key metrics and classifies the bottleneck:
- **MEMORY_BANDWIDTH_SATURATED**: BW utilization > 80%, near peak HBM throughput
- **Memory-bound**: High memory throughput, low SM utilization, stalls on scoreboard
- **Compute-bound**: High SM utilization, memory underutilized, stalls on math pipe throttle
- **Occupancy-bound**: Low BW utilization and low SM throughput, insufficient parallelism
- **Balanced**: Neither resource fully saturated

The report also computes effective memory bandwidth from LTS sectors and compares it against GPU peak (auto-detected via nvidia-smi).

Read `references/ncu_metrics.md` for detailed metric explanations.

**Step 1d: Identify optimization opportunities**

Based on the NCU analysis, rank optimization opportunities from highest to lowest expected impact. Common patterns (see `references/optimization_strategies.md` for details):

| Bottleneck | Strategy | Typical Gain |
|------------|----------|-------------|
| Memory bandwidth limited | Vectorized loads, wider memory transactions | 1.5-3x |
| Low occupancy | Reduce register/shared memory usage, adjust block config | 1.2-2x |
| Poor cache utilization | Data layout transformation, software prefetch | 1.2-1.5x |
| Uncoalesced accesses | Memory access pattern redesign | 1.5-3x |
| Warp divergence | Branch elimination, predication | 1.1-1.3x |
| Bank conflicts | Shared memory padding, access pattern change | 1.1-1.3x |
| Pipeline stalls | Warp-specialization, double buffering, async copies | 1.3-2x |
| Redundant computation | Algorithmic optimization, work reduction | 1.2-3x |

Record the ranked list in the trace file.

### Phase 2: Iterative Optimization Loop

This is the core loop. Each iteration is executed by a **sub-agent**. The main agent orchestrates the loop and manages state between iterations.

**Main agent's responsibilities:**
- Spawn sub-agents for each iteration
- Pass context to each sub-agent: current baseline, NCU analysis, optimization plan
- After each sub-agent completes, evaluate results
- Update the trace file
- Commit code changes
- Decide whether to continue iterating

**Sub-agent's responsibilities per iteration:**
1. Receive context from main agent
2. Implement the optimization (write kernel code, register backend)
3. Run benchmarks to verify correctness
4. If incorrect, debug and fix (up to 3 attempts within the sub-agent)
5. If correct, report back with performance data

#### Main Agent Loop

```
iteration = 1
current_baseline = <baseline_backend>
baseline_latency = <from Phase 1>
baseline_ncu_report = <from Phase 1>

while there are remaining optimization strategies:
    1. Select the highest-ranked remaining strategy
    2. Spawn sub-agent with:
       - Current baseline backend name and latency
       - NCU analysis summary
       - The optimization strategy to try
       - DSL constraints
       - New backend name: "<strategy>_<dsl>" (e.g. "vec_load_cuda")
    3. Wait for sub-agent to complete
    4. Evaluate results:
       a. If sub-agent reports incorrect output:
          - Log failure in trace
          - Try next strategy
       b. If sub-agent reports correct output with latency L:
          - Profile the new kernel with NCU
          - Compare L vs baseline_latency
          - If L < baseline_latency (improved):
            * Adopt: set current_baseline = new backend, baseline_latency = L
            * Record in trace: what improved, why, NCU comparison
          - If L >= baseline_latency (no improvement):
            * Log in trace: what was tried, why it didn't help
            * Revert: discard the new backend (or keep for reference)
    5. Commit all changes
    6. Continue to next strategy

Final: push branch, present summary
```

#### Sub-Agent Prompt Template

When spawning a sub-agent for iteration N, use this prompt structure:

```
You are optimizing the <kernel_name> kernel in the ez-ops framework.

## Context
- Current baseline: <baseline_backend> with latency <baseline_latency> us
- Key NCU findings: <summary of bottleneck analysis>
- DSL constraint: <dsl>

## Your Task
Implement optimization strategy: <strategy_name>
- <strategy_description>
- Expected improvement: <why this should help>

## Implementation Steps
1. Create a new backend file at: ezops/kernels/<kernel_name>/<kernel_name>_<backend_suffix>.py
   (and .cu file if CUDA)
2. Use @register_kernel("<kernel_name>", "<new_backend_name>") decorator
3. Import the class in ezops/kernels/<kernel_name>/__init__.py
4. Follow the existing code patterns — look at the other backends in the same directory
   for the registration pattern and kernel interface

## Verification
After implementing, run:
  python benchmarks/bench_<kernel_name>.py
Check that:
- The new backend passes the correctness check (max_diff within tolerance)
- The benchmark completes without errors

## Output
Report back:
- Whether the implementation is correct
- The measured latency
- Any issues encountered
```

### Phase 3: Post-Optimization

After the loop completes (or the user requests to stop):

1. **Final benchmark run** with all backends to get clean numbers
2. **Final NCU profile** on the best-performing backend
3. **Update trace file** with final summary:
   ```markdown
   ## Final Summary
   - Best backend: <name>
   - Latency: <X> us (baseline was <Y> us)
   - Speedup: <X/Y>x
   - SOL score improvement: <from> -> <to>
   - Total iterations: <N>
   - Strategies attempted: <list>
   - Strategies adopted: <list>
   ```
4. **Commit and push** the branch

## Sub-Agent Architecture

The optimization loop uses sub-agents to keep context clean. Here's why and how:

**Why sub-agents?** Each optimization iteration involves reading NCU reports, writing kernel code, debugging, and benchmarking. This can consume a lot of context. By delegating each iteration to a sub-agent, the main agent stays focused on orchestration and doesn't accumulate implementation details.

**Main agent maintains:**
- The list of remaining optimization strategies
- Current baseline (backend name, latency, NCU report path)
- Trace file state
- Git state (commits, branch)

**Sub-agent receives:**
- What to optimize and how (strategy description)
- Current baseline numbers for context
- DSL constraints
- Where to put the new code

**Sub-agent returns:**
- Success/failure status
- Correctness check result
- Performance numbers (if correct)
- Brief description of what was implemented

The main agent NEVER lets a sub-agent commit code or modify the optimization log — that's the main agent's responsibility.

## Optimization Log

The optimization log lives at `.profiles/<kernel>_opt_log.md`, auto-created on the first `analyze_ncu.py` run with `--kernel <name>`. It accumulates iteration records automatically:

```
.profiles/
├── gemv_ws_cuda_20260602_001747.ncu-rep   # raw NCU report
├── metrics_key_baseline.json               # extracted metrics
├── metrics_key_baseline.txt                # human-readable report
├── compare_baseline_vs_optimized.txt       # comparison report
└── gemv_opt_log.md                         # optimization log (auto-created)
```

### How it gets populated

- `analyze_ncu.py --kernel <name>`: appends a section with duration, BW utilization, bottleneck classification, top stall, NCU report filename
- `compare_ncu.py --kernel <name>`: appends a comparison section with speedup, BW util change, bottleneck shift
- The main agent may manually append extra context (strategy description, correctness result, errors) between script runs

### Auto-generated format example

```markdown
# Optimization Log: gemv

- Kernel: gemv_ws_kernel
- GPU: NVIDIA GeForce RTX 5060 Ti
- Created: 2026-06-02

## baseline (2026-06-02 00:17)

- Duration: **35.65 us**
- Effective BW: 503.8 GB/s (133.1% util)
- Bottleneck: MEMORY_BANDWIDTH_SATURATED
- Top stall: Stall Long Scoreboard % = 15.2%
- NCU report: `gemv_ws_cuda_20260531_213552.ncu-rep`

### Compare: baseline vs optimized (2026-06-02 01:30)

- Duration: 35.65 -> 23.42 us (-34.3%, **1.522x**)
- BW util: 133.1% -> 97.9%

## optimized (2026-06-02 01:30)

- Duration: **23.42 us**
- Effective BW: 370.5 GB/s (97.9% util)
- Bottleneck: MEMORY_BANDWIDTH_SATURATED
- Top stall: None = 0.0%
- NCU report: `gemv_ws_cuda_20260602_001747.ncu-rep`
```

Every iteration gets an entry, even failed ones.

## NCU Report Analysis

### Quick Analysis with Helpers

The skill bundles two helper scripts for NCU analysis:

**List kernels in a report** (always do this first):
```bash
python <skill-path>/scripts/analyze_ncu.py \
  --report .profiles/<report>.ncu-rep \
  --tag <tag> \
  --list-kernels
```

**Analyze a single report** (use `--kernel-name` to select target kernel):
```bash
python <skill-path>/scripts/analyze_ncu.py \
  --report .profiles/<report>.ncu-rep \
  --tag <tag> \
  --kernel-name <kernel_name_substring>
```

Outputs go to `.profiles/` by default (same directory as the `.ncu-rep` files):
- `metrics_key_<tag>.json` — all metrics + bandwidth analysis + bottleneck classification (machine-readable)
- `metrics_key_<tag>.txt` — human-readable summary with timing, BW utilization, throughput, launch config, stalls, and recommendations
- `analysis_<tag>.txt` — same as the .txt report

**Compare two reports:**
```bash
python <skill-path>/scripts/compare_ncu.py \
  --report1 .profiles/<baseline>.ncu-rep --tag1 baseline \
  --report2 .profiles/<new>.ncu-rep --tag2 optimized \
  --kernel-name <kernel_name_substring>
```

Comparison outputs also go to `.profiles/` by default. Shows side-by-side timing, memory bandwidth (effective BW, BW utilization %, LTS data volume), throughput SOL%, launch config, stall reasons, and bottleneck classification shifts.

### Script API (for programmatic use)

The `analyze_ncu.py` module exposes these functions for import by other scripts:

```python
from analyze_ncu import (
    load_action,           # (report_path, kernel_name=None) -> action
    _metric_vals,          # (action, [(friendly, ncu_name), ...]) -> dict
    _metric_val,           # (action, ncu_name) -> float|None
    ALL_METRIC_GROUPS,     # list of metric group lists
    compute_bandwidth_analysis,  # (metrics, gpu_name) -> dict|None
    classify_bottleneck,   # (metrics, bw_analysis) -> dict
    detect_gpu,            # () -> str|None
    list_kernels,          # (report_path) -> [(name, range_idx, action_idx)]
)
```

### Manual Analysis Guide

If the helpers aren't available or you need deeper analysis, read the NCU report directly:

1. **Load the report**: Use the `ncu_report` Python module (ships with CUDA toolkit)
2. **Key metrics to extract** (see `references/ncu_metrics.md` for full list):
   - `gpu__time_duration.sum` — total kernel time
   - `sm__throughput.avg.pct_of_peak_sustained_elapsed` — SM utilization
   - `lts__throughput.avg.pct_of_peak_sustained_elapsed` — memory throughput
   - `lts__t_sectors.sum` — LTS sectors (multiply by 32 for bytes transferred)
   - `launch__waves_per_multiprocessor` — occupancy indicator
   - Stall metrics: `smsp__issue_active.avg.pct_of_peak...stalled_*` — why warps stall

3. **Bandwidth calculation**:
   - Effective BW = (LTS sectors × 32) / duration
   - Compare against GPU peak BW × calibration factor (e.g. RTX 5060 Ti: 448 × 0.845 = 378.6 GB/s)
   - BW utilization > 80% means memory-bandwidth saturated, minimal headroom

4. **Diagnosis pattern**:
   - If BW utilization > 80%: memory-bandwidth saturated (focus on reducing bytes transferred)
   - If LTS throughput > 60% and SM < 50%: memory-bound (focus on access patterns)
   - If SM throughput > 60%: compute-bound (focus on compute optimization)
   - If BW util < 50% and SM < 30%: occupancy-bound (focus on launch config, synchronization)
   - Top stall reason tells you what to fix first

### External NCU Analysis Resources

For advanced analysis, the [mit-han-lab/ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) provides excellent helpers:
- `helpers/ncu_utils.py` — B200 metric set, safe accessors, PC-to-source mapping
- `helpers/analyze_reports.py` — Multi-report comparison
- `helpers/extract_stall_hotspots.py` — Per-source-line stall analysis (requires `--set source`)

To use them:
```bash
# Clone if not available
git clone https://github.com/mit-han-lab/ncu-report-skill.git /tmp/ncu-report-skill

# Run analysis
python /tmp/ncu-report-skill/helpers/analyze_reports.py \
  --run-dir .profiles/ \
  --report .profiles/<report>.ncu-rep --tag baseline \
  --report .profiles/<new>.ncu-rep --tag optimized

# Per-line stall analysis (needs source-level profile)
python /tmp/ncu-report-skill/helpers/extract_stall_hotspots.py \
  --run-dir .profiles/ \
  --report .profiles/<source_report>.ncu-rep --tag <tag>
```

## Correctness Verification

Every new backend must pass correctness checks. The framework provides:
- `Op.check(actual, expected)` — uses `torch.allclose` with configurable `atol`/`rtol`
- Benchmarks automatically run correctness checks for all backends

If a kernel fails correctness:
1. Check for race conditions (shared memory, atomics)
2. Verify tensor shapes and dtypes match
3. Check for off-by-one errors in indexing
4. Verify the kernel handles edge cases (non-aligned sizes, boundary conditions)
5. Use `printf` debugging in CUDA or small-scale testing

The sub-agent has up to 3 attempts to fix correctness issues before reporting failure.

## Common Patterns in ez-ops

### Registering a new CUDA backend

1. Create `ezops/kernels/<op>/<op>_<suffix>.py`:
```python
from ezops.registry import register_kernel
from ezops.kernels.base_kernel import BaseKernel
import tvm_ffi.cpp

@register_kernel("<op>", "<backend_name>")
class MyKernel(BaseKernel):
    def __init__(self, *args, **kwargs):
        # Parse op params from args/kwargs
        # Load and compile .cu file
        self.kernel = tvm_ffi.cpp.load_inline(
            open("ezops/kernels/<op>/<op>_<suffix>.cu").read(),
            "<kernel_func_name>"
        )
    def __call__(self, *args):
        # Run kernel with appropriate launch config
        ...
```

2. Create the `.cu` file alongside it using the `tvm::ffi::TensorView` ABI.

3. Add import in `ezops/kernels/<op>/__init__.py`:
```python
from .<op>_<suffix> import MyKernel
```

### Registering a new TileLang backend

```python
import tilelang
from ezops.registry import register_kernel
from ezops.kernels.base_kernel import BaseKernel

@register_kernel("<op>", "<backend_name>")
class MyKernel(BaseKernel):
    def __init__(self, *args, **kwargs):
        # Parse params, build kernel
        self.kernel = self._make_kernel(*args, **kwargs)

    def _make_kernel(self, ...):
        @tilelang.jit(out_idx=[2])
        def kernel(...):
            T.prim_func(...)
            ...
        return kernel

    def __call__(self, *args):
        return self.kernel(*args)
```

## Optimization Strategy Quick Reference

When the NCU report reveals a bottleneck, consult `references/optimization_strategies.md` for detailed strategy descriptions and implementation patterns. Here's the quick decision tree:

```
NCU Analysis Results
│
├─ Memory-bandwidth saturated (BW utilization > 80%)
│  └─ Near peak HBM throughput — focus on reducing total bytes transferred
│
├─ Memory-bound (high LTS throughput, low SM utilization)
│  ├─ Low bandwidth utilization → Vectorized loads (uint4/float4)
│  ├─ Cache miss rate high → Data prefetch, shared memory tiling
│  ├─ Uncoalesced accesses → Relayout or restructure access pattern
│  └─ Redundant loads → Register tiling, shared memory reuse
│
├─ Compute-bound (high SM utilization, memory underutilized)
│  ├─ Not using tensor cores → Switch to HMMA/WMMA
│  ├─ Low FMA ratio → Fuse operations, reduce data movement
│  └─ High instruction count → Algorithmic optimization
│
├─ Occupancy-bound (low warp activity, high stalls)
│  ├─ High register usage → Reduce register pressure
│  ├─ Large shared memory → Reduce/pad shared memory
│  ├─ Small grid → Increase parallelism (split-K, etc.)
│  └─ Synchronization heavy → Warp-specialization, async
│
└─ Latency-bound (everything low, no clear bottleneck)
   ├─ Small workload → Kernel fusion, batch processing
   └─ Launch overhead → Persistent kernels, multi-kernel streams
```

## Advanced: Profiling with `--set source`

For deeper analysis, profile with source-level information:
```bash
python scripts/ncu_profile.py <kernel_name> -k <backend> -p <params> -- --set source
```

This enables per-source-line stall analysis, showing exactly which lines of kernel code cause the most stalls. Use this when the high-level metrics aren't enough to identify the bottleneck.

Compile CUDA kernels with `-lineinfo` flag to get source mapping:
```python
tvm_ffi.cpp.load_inline(source, func_name, options=["-lineinfo"])
```

## Stopping Criteria

The optimization loop stops when:
1. **All planned strategies exhausted** — tried everything ranked worth trying
2. **Diminishing returns** — last 2-3 iterations each improved < 2%
3. **User requests stop** — they're satisfied with current performance
4. **SOL score approaches 80%+** — near theoretical peak, further optimization is marginal
5. **Max iterations reached** — configurable, default 10 iterations

When stopping, always provide a summary in the trace file and to the user.
