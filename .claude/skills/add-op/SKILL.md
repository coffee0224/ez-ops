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

    def __init__(self, <params>, backend: str = "triton"):
        # Store all problem-size parameters
        self.<params> = <params>
        self._backend = backend
        kernel_cls = get_kernel("<op_name>", backend)
        self._kernel = kernel_cls(<params>)

    def forward(self, <tensor_args>) -> <return_annotation>:
        self._kernel(<tensor_args>)

    def gen_data(self):
        # Create CUDA float32 tensors with appropriate shapes
        ...
        return <tuple_of_tensors>

    def _ref_forward(self, <tensor_args>) -> <return_annotation>:
        # Pure PyTorch reference — this is the "correct" implementation
        ...
```

Key conventions:
- `_params_desc` is a class-level dict mapping constructor parameter names to human-readable descriptions. Used by `scripts/ncu_profile.py -h` to show parameter info. Fill in a one-line description for each problem-size parameter.
- Constructor stores problem-size params and instantiates the kernel via `get_kernel`.
- `forward` delegates entirely to `self._kernel(...)`.
- `gen_data` returns a tuple of CUDA float32 tensors. Output tensors use `torch.empty`.
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

Create `benchmarks/bench_<op_name>.py` with two separate functions:
`bench_correctness` for correctness checking and `bench_latency` for performance measurement.

```python
import torch

from ezops import <PascalCase>Op, list_backends
from ezops.ops.utils.bench import bench_kernel

BACKENDS = list_backends("<op_name>")
<PROBLEM_SIZE_CONFIG>
WARMUP = 10
N_REPEAT = 50
N_TRIALS = 3


def bench_correctness():
    ref_op = <PascalCase>Op(<params>, backend="triton")
    <data> = ref_op.gen_data()
    ref_op._ref_forward(*<data>)

    print(f"{'backend':<12} {'max_diff':>12} {'result':>10}")
    print("-" * 36)

    for backend in BACKENDS:
        try:
            op = <PascalCase>Op(<params>, backend=backend)
        except Exception as e:
            print(f"{backend:<12} {'—':>12} {'ERROR':>10}  {e}")
            continue

        <fresh_data> = op.gen_data()
        op(*<input_data>)

        max_diff = (<output> - <ref_output>).abs().max().item()
        passed = op.check(<output>, <ref_output>)
        print(f"{backend:<12} {max_diff:>12.2e} {'PASS' if passed else 'FAIL':>10}")


def bench_latency():
    ref_op = <PascalCase>Op(<params>, backend="triton")
    <data> = ref_op.gen_data()

    print(f"{'backend':<12} {'latency_ms':>12}")
    print("-" * 26)

    for backend in BACKENDS:
        try:
            op = <PascalCase>Op(<params>, backend=backend)
        except Exception as e:
            print(f"{backend:<12} {'ERROR':>12}  {e}")
            continue

        ms = bench_kernel(op, args=<data_tuple>, n_warmup=WARMUP, n_repeat=N_REPEAT, n_trials=N_TRIALS)
        print(f"{backend:<12} {ms:>12.4f}")


if __name__ == "__main__":
    bench_correctness()
    print()
    bench_latency()
```

Key conventions for the benchmark:
- **Correctness** uses `op.check(output, ref_output)` which applies `torch.allclose` with the op's own `_atol` / `_rtol` defaults (inherited from `Op` base class: `1e-6` / `1e-5`). Ops can override these in `__init__` if needed.
- **Latency** uses `bench_kernel` from `ezops.ops.utils.bench` which follows the NVIDIA SOL-ExecBench protocol: L2 cache flush before every iteration, input tensor clone pool to avoid cache effects, multiple independent trials with median selection.
- Adapt the data unpacking and comparison to match the op's tensor signature.
- For ops that write in-place, compare the output tensor against `C_ref`.
- For ops that return a value, compare the return value.

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
