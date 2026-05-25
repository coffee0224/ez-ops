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


