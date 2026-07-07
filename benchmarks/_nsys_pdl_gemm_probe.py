"""Standalone eager probe for nsys profiling of pdl_gemm.

Runs each backend once on the tiny workload, no L2 flush, no graph capture,
so the timeline shows the raw launch + execute of the two GEMM kernels.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ezops import PdlGemmOp

M, K, N, P = 64, 128, 128, 128

torch.manual_seed(42)
ref_op = PdlGemmOp(M, K, N, P, backend="ref")
x, W1, W2, y, z = ref_op.gen_data()
ref_op._ref_forward(x, W1, W2, y, z)
z_ref = z.clone()

# Warmup each backend so JIT / driver setup cost is not on the captured path.
for backend in ["cuda", "cuda_pdl"]:
    op = PdlGemmOp(M, K, N, P, backend=backend)
    y_b = torch.empty_like(y)
    z_b = torch.empty_like(z)
    for _ in range(20):
        op(x, W1, W2, y_b, z_b)
torch.cuda.synchronize()

# Now run each backend a few times with named ranges around it so nsys shows
# clean per-backend rows in the timeline.
for backend in ["cuda", "cuda_pdl"]:
    op = PdlGemmOp(M, K, N, P, backend=backend)
    y_b = torch.empty_like(y)
    z_b = torch.empty_like(z)
    for rep in range(5):
        torch.cuda.nvtx.range_push(f"{backend}#{rep}")
        op(x, W1, W2, y_b, z_b)
        torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
