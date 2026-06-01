# ez-ops Backend Registration Guide

How to add new kernel backends to the ez-ops framework.

## Quick Start

Adding a new backend involves 3 steps:
1. Create the kernel implementation file(s)
2. Register with `@register_kernel`
3. Import in the op's `__init__.py`

## Step 1: Create Kernel Files

Backend files go in `ezops/kernels/<op_name>/`. Name convention: `<op>_<backend>.py` (and optionally `<op>_<backend>.cu` for CUDA).

### CUDA Backend Template

Create `<op>_<backend>.py`:

```python
from ezops.registry import register_kernel
from ezops.kernels.base_kernel import BaseKernel
import tvm_ffi.cpp

@register_kernel("<op_name>", "<backend_name>")
class MyCUDAKernel(BaseKernel):
    def __init__(self, M, N, K, **kwargs):
        self.M = M
        self.N = N
        self.K = K
        # Load CUDA source
        cu_source = open(f"ezops/kernels/<op_name>/<op>_<backend>.cu").read()
        self.kernel_func = tvm_ffi.cpp.load_inline(
            cu_source,
            "kernel_func_name",
            options=["-lineinfo"],  # Include for NCU source-level profiling
        )

    def __call__(self, A, B, C):
        # Set up launch configuration
        grid = (..., 1, 1)
        block = (..., 1, 1)
        # Launch kernel
        self.kernel_func(A, B, C, self.M, self.N, self.K, grid=grid, block=block)
        return C
```

Create `<op>_<backend>.cu` alongside:

```cuda
#include <tvm_ffi.h>

__global__ void kernel_func_name(
    tvm_ffi::TensorView<float, 2> A,
    tvm_ffi::TensorView<float, 2> B,
    tvm_ffi::TensorView<float, 2> C,
    int M, int N, int K) {
    // Kernel implementation
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // ...
}

// Note: tvm_ffi.cpp.load_inline uses the first __global__ function
// or you specify the name explicitly
```

### TileLang Backend Template

```python
import tilelang
from ezops.registry import register_kernel
from ezops.kernels.base_kernel import BaseKernel

@register_kernel("<op_name>", "<backend_name>")
class MyTileLangKernel(BaseKernel):
    def __init__(self, M, N, K, **kwargs):
        self.M = M
        self.N = N
        self.K = K
        self.kernel = self._make_kernel(M, N, K)

    def _make_kernel(self, M, N, K):
        @tilelang.jit(out_idx=[2])
        def kernel(A, B, C):
            T.prim_func(A, B, C)
            # TileLang kernel definition
            # ...
        return kernel

    def __call__(self, A, B, C):
        return self.kernel(A, B, C)
```

### Triton Backend Template

```python
import torch
import triton
import triton.language as tl
from ezops.registry import register_kernel
from ezops.kernels.base_kernel import BaseKernel

@triton.jit
def _kernel_func(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_bk, stride_bn, stride_cm,
    BLOCK_SIZE: tl.constexpr,
):
    # Triton kernel implementation
    pid = tl.program_id(0)
    # ...

@register_kernel("<op_name>", "<backend_name>")
class MyTritonKernel(BaseKernel):
    def __init__(self, M, N, K, **kwargs):
        self.M = M
        self.N = N
        self.K = K

    def __call__(self, A, B, C):
        grid = (self.M, 1, 1)
        _kernel_func[grid](
            A, B, C,
            self.M, self.N, self.K,
            A.stride(0), B.stride(0), B.stride(1), C.stride(0),
            BLOCK_SIZE=256,
        )
        return C
```

## Step 2: Register

The `@register_kernel(op_name, backend_name)` decorator handles registration. Choose a descriptive backend name:

- `cuda` — standard CUDA
- `ws_cuda` — warp-specialized CUDA
- `<feature>_<dsl>` — e.g., `vec_load_cuda`, `splitk_tilelang`
- `<approach>_<dsl>` — e.g., `naive_gemv_tilelang`, `autotune_tilelang`

Names should be lowercase with underscores. They appear in benchmark output and NCU profiling commands.

## Step 3: Import

Add the import in `ezops/kernels/<op_name>/__init__.py`:

```python
from .<op>_<backend> import MyKernel
```

This ensures the decorator fires when the package is imported.

## Verification

After registration, verify with:

```bash
# Check it shows up
python -c "from ezops import list_backends; print(list_backends('<op_name>'))"

# Run benchmarks
python benchmarks/bench_<op_name>.py

# Profile with NCU
python scripts/ncu_profile.py <op_name> -k <backend_name> -p <params>
```

## Common Patterns from Existing Backends

### GEMV CUDA (gemv_cu.py)
- Uses shared memory for vector A with bank-conflict padding
- Each warp computes one output row
- Vectorized `uint4` loads for matrix B

### GEMV Warp-Specialized (gemv_ws_cu.py)
- 1 DMA warp + N compute warps per block
- TMA for matrix B loading (SM90+)
- mbarrier for producer-consumer synchronization
- 2-stage double buffering

### GEMV TileLang (gemv_tl.py)
- Multiple variants with different tiling strategies
- Uses `tilelang.autotuner.autotune` for config search
- `alloc_reducer` for distributed accumulation

### Attention Decode CUDA (attn_decode_cu.py)
- One block per (batch, head)
- Warps iterate KV in parallel
- Online softmax for numerical stability
- vec4 loads for Q/K/V
