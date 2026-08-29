---
name: add-op
description: >
  Add a new operator (op) to the ez-ops project. Use this skill whenever the user wants to
  add, create, or define a new op/operator — including named ops like "vector_add", "matmul",
  "softmax", "relu", "conv2d", etc. Also triggers when the user says "add a new op", "create
  an operator", "implement op X", or describes a computation they want to turn into an op.
  This skill scaffolds the op class, kernel templates (triton/tilelang/cuda), registrations,
  imports, and benchmark script in one pass.
---

# Add a New Op to ez-ops

You are adding a new operator to the ez-ops project. This involves creating files across
four areas: the op definition, kernel templates, import registrations, and a benchmark script.

Before writing any code, ask the user for:
1. **Op name** (snake_case, e.g. `vector_add`, `matmul`, `softmax`) — used for file names and registry keys.
2. **Computation description** — what the op does mathematically (e.g. "C = A + B", "Y = softmax(X)", "out = A @ B").
3. **Tensor signatures** — input and output tensor shapes, dtypes. If the user doesn't specify, infer reasonable defaults from the computation.

If the user's request already contains all of this information, proceed without asking.

## Step 1: Create the Op class

Create `ezops/ops/<op_name>.py` following the VectorAddOp pattern exactly:

```python
import torch

from .base_op import Op
from ..registry import get_kernel


class <PascalCase>Op(Op):
    _params_desc = {"<param1>": "<description>", "<param2>": "<description>"}

    def __init__(self, <params>, backend: str = "ref"):
        self.<params> = <params>
        self._backend = backend
        if backend != "ref":
            kernel_cls = get_kernel("<op_name>", backend)
            self._kernel = kernel_cls(<params>)
        else:
            self._kernel = self._ref_forward

    def forward(self, <tensor_args>) -> <return_annotation>:
        self._kernel(<tensor_args>)

    def gen_data(self):
        ...
        return <tuple_of_tensors>

    def _ref_forward(self, <tensor_args>) -> <return_annotation>:
        ...
```

Key conventions:
- `_params_desc` is a class-level dict mapping constructor parameter names to human-readable descriptions. Used by `scripts/ncu_profile.py -h` to show parameter info. Fill in a one-line description for each problem-size parameter.
- Default `backend` is `"ref"`. When `"ref"`, `self._kernel` is set to `self._ref_forward` so `forward()` calls the PyTorch reference directly. When any other backend, `self._kernel` is instantiated via `get_kernel`.
- `forward` delegates entirely to `self._kernel(...)`.
- `gen_data` returns a tuple of CUDA tensors. Output tensors use `torch.empty`.
- `_ref_forward` is a plain PyTorch implementation — no custom kernels, no Triton.
- If the op writes output in-place into an existing tensor (like VectorAddOp), `_ref_forward` returns `None` and uses `C.copy_(...)` or similar.
- If the op returns a new tensor, `_ref_forward` returns it directly.
- Correctness tolerance is controlled by `self._atol` and `self._rtol` (defaults `1e-6` / `1e-5` from the `Op` base class). The base class provides `op.check(actual, expected)` which calls `torch.allclose` with these values. Override in `__init__` if the op needs different tolerances.

## Step 2: Create kernel templates

Create the directory `ezops/kernels/<op_name>/` with these four files:

### 2a: `ezops/kernels/<op_name>/__init__.py`

```python
from .<op_name>_triton import <PascalCase>TritonKernel  # noqa: F401
from .<op_name>_tl import <PascalCase>TileLangKernel  # noqa: F401
from .<op_name>_cu import <PascalCase>CudaKernel  # noqa: F401
```

### 2b: `ezops/kernels/<op_name>/<op_name>_triton.py`

```python
import triton
import triton.language as tl

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("<op_name>", "triton")
class <PascalCase>TritonKernel(BaseKernel):
    def __init__(self, <params>):
        self.<params> = <params>
        self.block_size = 1024  # TODO: tune for this op

    @staticmethod
    @triton.jit
    def _kernel(<pointers_and_params>, BLOCK_SIZE: tl.constexpr):
        # TODO: implement the triton kernel for <op_name>
        raise NotImplementedError("TODO: implement triton kernel for <op_name>")

    def __call__(self, <tensor_args>) -> None:
        assert <shape_checks>
        grid = lambda meta: (<grid_size>,)
        self._kernel[grid](<args>, BLOCK_SIZE=self.block_size)
```

Write the full skeleton with correct grid computation, argument passing, and shape assertions.
Mark the kernel body with `# TODO` and `raise NotImplementedError` — the user will fill in the actual compute logic.

### 2c: `ezops/kernels/<op_name>/<op_name>_tl.py`

```python
from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("<op_name>", "tilelang")
class <PascalCase>TileLangKernel(BaseKernel):
    def __init__(self, <params>):
        self.<params> = <params>
        self.block_size = 1024  # TODO: tune for this op
        self._program = self._build()

    def _build(self):
        # TODO: implement the tilelang kernel for <op_name>
        raise NotImplementedError("TODO: implement tilelang kernel for <op_name>")

    def __call__(self, <tensor_args>) -> None:
        # TODO: implement the call logic for <op_name>
        raise NotImplementedError("TODO: implement tilelang __call__ for <op_name>")
```

### 2d: `ezops/kernels/<op_name>/<op_name>_cu.py`

```python
from pathlib import Path

from tvm_ffi import cpp

from ..base_kernel import BaseKernel
from ...registry import register_kernel

_CU_SRC = Path(__file__).with_suffix(".cu")


@register_kernel("<op_name>", "cuda")
class <PascalCase>CudaKernel(BaseKernel):
    def __init__(self, <params>):
        self.<params> = <params>
        self._mod = cpp.load_inline(
            name="<op_name>_cuda",
            cuda_sources=_CU_SRC.read_text(),
            functions="<op_name>_cu",
            extra_cuda_cflags=["-O3", "--generate-line-info"],
        )

    def __call__(self, <tensor_args>) -> None:
        # TODO: implement the cuda call for <op_name>
        raise NotImplementedError("TODO: implement cuda __call__ for <op_name>")
```

### 2e: `ezops/kernels/<op_name>/<op_name>_cu.cu`

```cpp
#include <tvm/ffi/tvm_ffi.h>

// TODO: implement the CUDA kernel for <op_name>
__global__ void <op_name>_kernel(<params>) {
    // TODO: kernel body
}

void <op_name>_cu(tvm::ffi::TensorView <args>) {
    // TODO: host launch logic
    // Use TVMFFIEnvGetStream for the CUDA stream:
    //   DLDevice dev = A.device();
    //   cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
    raise NotImplementedError("TODO: implement CUDA kernel for <op_name>")
}
```

Write the full `.cu` skeleton with correct function signatures, grid/block setup, and stream handling.
Leave the actual kernel body as `// TODO`.

## Step 3: Update imports

Three `__init__.py` files need updating:

### 3a: `ezops/kernels/__init__.py`

Add the new kernel subpackage import:
```python
from . import <op_name>  # noqa: F401
```

### 3b: `ezops/ops/__init__.py`

Add the new op export:
```python
from .<op_name> import <PascalCase>Op
```
And append `"<PascalCase>Op"` to `__all__`.

### 3c: `ezops/__init__.py`

Add the top-level export:
```python
from .ops.<op_name> import <PascalCase>Op
```
And append `"<PascalCase>Op"` to `__all__`.

The import chain must be: `ezops/__init__.py` triggers `kernels/__init__.py` which triggers each kernel subpackage's `__init__.py` which imports kernel modules causing `@register_kernel` decorators to fire. This chain is critical — without it, no kernels are registered and `get_kernel` will fail.

## Step 4: Create the benchmark script

Create `benchmarks/bench_<op_name>.py` with correctness, latency, and SOL analysis.

The benchmark uses a **workload** pattern: define a `WORKLOADS` list of parameter tuples with labels,
then iterate over each workload, running all backends and printing per-workload tables.

```python
import argparse
import math
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from tabulate import tabulate

from ezops import <PascalCase>Op, list_backends
from ezops.ops.utils.bench import bench_kernel
from ezops.ops.utils.accuracy import SQNR_THRESHOLD_DB, check_determinism, check_input_readonly, sqnr_db

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks.hardware.gpu_specs import detect_profile

BACKENDS = list_backends("<op_name>")
print(BACKENDS)
WARMUP = 10
N_REPEAT = 50
N_TRIALS = 3

WORKLOADS = [
    # (<param1>, <param2>, ..., "label"),
    # Fill in representative workloads, e.g. from real model layer shapes.
]


def _detect_gpu_profile():
    """Detect current GPU via nvidia-smi and return profile key."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return None
        name = r.stdout.strip().split("\n")[0].strip()
        return detect_profile(name)
    except FileNotFoundError:
        return None


def _load_profile(profile_name):
    """Load GPU profile YAML from assets/."""
    path = ROOT / "assets" / f"{profile_name}.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _compute_sol(roofline, profile, input_bytes):
    """Compute SOL metrics from roofline result and GPU profile.

    Uses tf32 tensor core peak as the FP32 compute ceiling.
    """
    hbm = profile.get("hbm", {})
    tc = profile.get("tensor_core", {}).get("tf32")
    if not hbm.get("theoretical") or not tc or not tc.get("theoretical"):
        return None

    eff_hbm = float(hbm["theoretical"]) * float(hbm.get("calibration", 1.0))
    eff_compute = float(tc["theoretical"]) * float(tc.get("calibration", 1.0))

    compute_t = roofline.flops / eff_compute
    mem_fused_t = input_bytes / eff_hbm
    mem_unfused_t = roofline.bytes / eff_hbm
    theo_min = max(compute_t, mem_fused_t)

    return {
        "compute_us": compute_t * 1e6,
        "mem_fused_us": mem_fused_t * 1e6,
        "mem_unfused_us": mem_unfused_t * 1e6,
        "theo_min_us": theo_min * 1e6,
        "theo_min_s": theo_min,
    }


def _fmt_sqnr(v: float) -> str:
    return "inf" if math.isinf(v) else f"{v:.1f}"


def _run_workload(<params>, label, profile, backends):
    """Run all backends for a single workload and print two tables.

    Accuracy phase mirrors the xpuoj judge order (all untimed): one call
    checked against the ref via SQNR + allclose, then determinism (two
    calls, byte-for-byte output compare), then input read-only.
    """
    print(f"\n{'=' * 60}")
    print(f"  <OP_DISPLAY> workload: {label}  (<param_summary>)")
    print(f"  <tensor_shape_summary>")
    print(f"  pass = allclose AND sqnr >= {SQNR_THRESHOLD_DB:.0f} dB; det = byte-identical over 2 calls")
    print(f"{'=' * 60}\n")

    torch.manual_seed(42)
    # Generate data for reference
    ref_op = <PascalCase>Op(<params>, backend="ref")
    <data> = ref_op.gen_data()
    ref_op._ref_forward(*<data>)
    roofline = ref_op.get_roofline()

    ref_ms = bench_kernel(
        ref_op._ref_forward, args=<data_tuple>,
        n_warmup=WARMUP, n_repeat=N_REPEAT, n_trials=N_TRIALS,
    )

    sol = _compute_sol(roofline, profile, <input_bytes>) if profile else None

    rows = []
    for backend in backends:
        try:
            op = <PascalCase>Op(<params>, backend=backend)
            <fresh_output> = <clone_or_empty>
            op(<input_args>, <output_arg>)
            max_diff = (<output> - <ref_output>).abs().max().item()
            sqnr = sqnr_db(<ref_output>, <output>)
            passed = op.check(<output>, <ref_output>) and sqnr >= SQNR_THRESHOLD_DB
            det_ok = check_determinism(op, inputs=(<input_args>,), outputs=<output_arg>)
            ro_ok = check_input_readonly(op, inputs=(<input_args>,), outputs=<output_arg>)
            ms = bench_kernel(op, args=(<bench_args>), n_warmup=WARMUP, n_repeat=N_REPEAT, n_trials=N_TRIALS)
            speedup = ref_ms / ms if ms > 0 else float("inf")
            sol_score = sol["theo_min_s"] / (ms / 1000) if sol else None
            rows.append([
                backend, f"{max_diff:.2e}", _fmt_sqnr(sqnr),
                "PASS" if passed else "FAIL",
                "OK" if det_ok else "NONDET",
                "OK" if ro_ok else "MUTATED",
                f"{ms:.4f}", f"{speedup:.2f}x",
                f"{sol_score:.1f}x" if sol_score is not None else "—",
            ])
        except Exception as e:
            print(e)
            continue

    ref_sol = sol["theo_min_s"] / (ref_ms / 1000) if sol else None
    rows.append([
        "ref", "—", "—", "—", "—", "—", f"{ref_ms:.4f}", "1.00x",
        f"{ref_sol:.1f}x" if ref_sol is not None else "—",
    ])

    # Table 1: Performance
    print(tabulate(
        rows,
        headers=["backend", "max_diff", "sqnr(dB)", "result", "det", "input", "latency(ms)", "speedup", "sol-score"],
        tablefmt="github",
    ))

    # Table 2: SOL analysis
    if sol:
        print()
        sol_rows = [
            ["compute time (tf32 peak)", f"{sol['compute_us']:.3f} µs"],
            ["mem time (fused)", f"{sol['mem_fused_us']:.3f} µs"],
            ["mem time (unfused)", f"{sol['mem_unfused_us']:.3f} µs"],
            ["theoretical min", f"{sol['theo_min_us']:.3f} µs"],
        ]
        print(tabulate(sol_rows, headers=["SOL metric", "value"], tablefmt="github"))


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False, description="<OP_DISPLAY> kernel benchmark")
    parser.add_argument("-h", action="store_true", dest="list_backends",
                        help="List available backends and exit")
    parser.add_argument("-k", "--backends", type=str, default=None,
                        help="Comma-separated list of backends to benchmark (ref is always included)")
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.list_backends:
        print("Available backends:", ", ".join(BACKENDS))
        return

    if args.backends is not None:
        selected = [b.strip() for b in args.backends.split(",")]
        invalid = [b for b in selected if b != "ref" and b not in BACKENDS]
        if invalid:
            print(f"Unknown backends: {', '.join(invalid)}")
            print(f"Available: {', '.join(BACKENDS)}")
            return
        backends = [b for b in selected if b in BACKENDS]
    else:
        backends = BACKENDS

    profile_name = _detect_gpu_profile()
    profile = _load_profile(profile_name) if profile_name else None

    for <params>, label in WORKLOADS:
        _run_workload(<params>, label, profile, backends)


if __name__ == "__main__":
    main()
```

Key conventions for the benchmark:
- **Workload pattern**: `WORKLOADS` is a list of tuples `(param1, param2, ..., "label")`. Each tuple contains the op constructor parameters followed by a human-readable label (e.g. model layer name). `main()` iterates over workloads and calls `_run_workload` for each.
- Each workload gets its own performance table and SOL table, separated by a header banner showing the workload label and parameter summary.
- Main table columns: `backend`, `max_diff`, `sqnr(dB)`, `result`, `det`, `input`, `latency(ms)`, `speedup`, `sol-score`.
- `ref` row at the bottom shows PyTorch reference latency as the speedup baseline (1.00x).
- The try/except covers the entire backend run; unimplemented backends print the exception and are skipped (no output row).
- **Accuracy phase mirrors the xpuoj judge order** (all untimed, before any warmup/timing), using helpers from `ezops.ops.utils.accuracy`:
  1. **SQNR vs ref** (one call): `sqnr_db(ref, out) = 10·log10(‖ref‖² / ‖ref-out‖²)` in dB. `inf` means bitwise-identical to ref. Threshold `SQNR_THRESHOLD_DB = 28.0` (xpuoj convention): healthy fp32/bf16 kernels land ≥ 50 dB; single-digit dB means a structural bug (bad indexing, misaligned copy), not rounding noise.
  2. **Determinism** (`check_determinism`): two calls on the same input, outputs compared **byte-for-byte** (uint8 view — distinguishes ±0.0 and NaN payloads). Cross-block `atomic_add` accumulation or unordered reductions show up as `NONDET`; single-block / fixed-order reductions are `OK`. Nondeterministic kernels fail the xpuoj judge, so `NONDET` is a real finding to fix (e.g. two-stage reduction), not a flaky check.
  3. **Input read-only** (`check_input_readonly`): input tensors must stay byte-identical after a call (`OK` / `MUTATED`).
- `passed` combines `op.check(output, ref_output)` (per-element `torch.allclose` with the op's `_atol` / `_rtol`, defaults `1e-6` / `1e-5` from the `Op` base class; ops can override in `__init__`) **AND** `sqnr >= SQNR_THRESHOLD_DB`. The workload banner states this pass criterion.
- Adapt `inputs=(...)` / `outputs=...` in the two check calls to the op's tensor signature: all input tensors go to `inputs`, all kernel-written tensors (in-place outputs, workspaces) go to `outputs`.
- **Latency** uses `bench_kernel` from `ezops.ops.utils.bench` which follows the NVIDIA SOL-ExecBench protocol: L2 cache flush before every iteration, input tensor clone pool to avoid cache effects, multiple independent trials with median selection.
- **SOL (Speed of Light)** analysis:
  - `sol-score = theoretical_min / latency` — upper bound is 1.0 (100% of theoretical peak).
  - `theoretical_min = max(compute_time, mem_time_fused)` — the roofline lower bound.
  - `compute_time = flops / effective_tf32_peak` where effective = theoretical × calibration from profile YAML.
  - `mem_time_fused = input_bytes / effective_hbm_bw` (only input tensors, output write assumed free).
  - `mem_time_unfused = total_bytes / effective_hbm_bw` (all tensor traffic).
  - GPU profile loaded from `assets/{profile}.yaml` (generated by `scripts/roofline_profile.py`).
  - If no matching hardware profile is found, `sol-score` shows "—" and SOL table is skipped.
- Adapt the data generation, unpacking, and comparison to match the op's tensor signature.
- For ops that write in-place, compare the output tensor against a cloned `C_ref`.
- For ops that return a value, compare the return value.
- Fill `WORKLOADS` with representative shapes (e.g. from real model layer dimensions) as a starting point.

## Execution order

Execute these steps in order:
1. Create `ezops/ops/<op_name>.py` (the op class)
2. Create `ezops/kernels/<op_name>/` directory with all four kernel files + `.cu`
3. Update the three `__init__.py` files
4. Create `benchmarks/bench_<op_name>.py`

After creating all files, verify the import chain works by running:
```bash
uv run python -c "from ezops import <PascalCase>Op; print('OK')"
```

And verify the roofline works:
```bash
uv run python -c "from ezops import <PascalCase>Op; op = <PascalCase>Op(<small_params>); print(op.get_roofline())"
```

Once the kernel backends are implemented, run the benchmark and confirm the accuracy
columns render and report honestly — `sqnr(dB)` well above 28, `det` OK, `input` OK:
```bash
uv run python benchmarks/bench_<op_name>.py -k <backend>
```
`NONDET` or `MUTATED` are findings to fix in the kernel (e.g. replace cross-block
atomics with a two-stage reduction), not harness false positives — the harness itself
is validated: the ref backend always reports `det OK`.
