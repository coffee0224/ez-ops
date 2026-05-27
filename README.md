# ez-ops

An opinionated framework for GPU operator development, benchmarking, and profiling.

Writing a high-performance GPU kernel is only half the battle — you also need to define workloads, compute theoretical ceilings, verify correctness, profile at SM granularity, and integrate with real models. **ez-ops** ties all of these into a single, coherent workflow.

## Features

- **Operator Definition** — Declarative spec for shapes, dtypes, and semantics. Define once, reuse everywhere.
- **Workload Definition** — Parametric workload generators with sweep/batch support.
- **SOL Analysis** — Auto-compute Arithmetic Intensity + hardware spec → Speed-of-Light roofline ceiling.
- **Correctness Checking** — Automatic numerical verification against reference implementations.
- **SM-Level Profiling** — Instrument kernels at SM/warp granularity, export to Perfetto / chrome://tracing.
- **Multi-Backend** — Write hardware-independent logic once; swap backends (CUDA, Triton, TileLang).
- **Cross-Implementation Benchmarking** — Compare your kernels against each other and against mature libraries (e.g., attention vs flash-attn, flashinfer; matmul vs cuBLAS/CUTLASS) under identical workloads.
- **Model Integration** — Hook operators into real model / inference frameworks for end-to-end testing.

## Quick Start

### 1. Generate a GPU Profile

Before benchmarking, generate a hardware profile for your GPU. This runs HBM bandwidth and matmul peak microbenchmarks, then writes a YAML file to `assets/`.

```bash
# Full calibration (recommended for first run)
python scripts/roofline_profile.py

# Quick mode (fewer iterations, smaller sweep)
python scripts/roofline_profile.py --quick

# Select dtypes and output directory
python scripts/roofline_profile.py --dtypes bf16,fp16,tf32,fp8 --output-dir assets/
```

This produces `assets/<gpu_profile>.yaml` (e.g. `assets/rtx_5060_ti.yaml`) containing theoretical peaks and measured calibration factors for HBM bandwidth and tensor core FLOPS.

### 2. Benchmark an Operator

Each operator has a benchmark script under `benchmarks/`:

```bash
python benchmarks/bench_vector_add.py
python benchmarks/bench_gemv.py
```

Output includes:

| backend | max_diff | result | latency(ms) | speedup | sol-score |
|---------|----------|--------|-------------|---------|-----------|
| triton  | 0.00e+00 | PASS   |      0.0312 | 1.50x   | 0.7x      |
| ref     | —        | —      |      0.0468 | 1.00x   | 0.5x      |

| SOL metric               | value     |
|--------------------------|-----------|
| compute time (tf32 peak) | 0.043 µs  |
| mem time (fused)         | 22.133 µs |
| mem time (unfused)       | 33.200 µs |
| theoretical min          | 22.133 µs |

- **sol-score** = theoretical_min / latency (1.0 = 100% of theoretical peak).
- **theoretical_min** = max(compute_time, mem_time_fused) — the roofline lower bound.
- If no GPU profile is found in `assets/`, sol-score is omitted and the SOL table is not printed.
