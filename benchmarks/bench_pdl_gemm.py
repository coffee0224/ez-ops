"""Benchmark PDL on two chained GEMMs.

Compares:
  * ref       — PyTorch eager matmul (two separate cuda launches)
  * cuda      — custom GEMM kernel, normal launch (no PDL)
  * cuda_pdl  — same GEMM kernel, launched with PDL launch attribute and
                griddepcontrol PTX inside the kernel

The interesting number is the speedup of `cuda_pdl` over `cuda` — that's
the latency PDL saves by overlapping FC2's prolog with FC1's epilogue.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from tabulate import tabulate

from ezops import PdlGemmOp, list_backends
from ezops.ops.utils.bench import bench_kernel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BACKENDS = list_backends("pdl_gemm")
WARMUP = 30
N_REPEAT = 200
N_TRIALS = 10

# Each workload is sized to keep each GEMM short — that's where PDL shines,
# since the ~3–5 us of overlap it saves is a meaningful fraction of total
# latency. Larger / compute-bound kernels see little benefit.
WORKLOADS = [
    # (M, K, N, P, label)
    (64,  128,  128,  128, "tiny-mlp"),
    (64,  512,  512,  512, "small-mlp"),
    (128, 1024, 1024, 1024, "medium-mlp"),
    (256, 2048, 2048, 2048, "large-mlp"),
]


def _run_workload(M, K, N, P, label, backends):
    print(f"\n{'=' * 70}")
    print(f"  PDL GEMM workload: {label}")
    print(f"  x: ({M}, {K})  W1: ({K}, {N})  W2: ({N}, {P})  -> z: ({M}, {P})")
    print(f"  grids: FC1 = {(N+63)//64}x{(M+63)//64}, FC2 = {(P+63)//64}x{(M+63)//64}")
    print(f"{'=' * 70}\n")

    torch.manual_seed(42)
    ref_op = PdlGemmOp(M, K, N, P, backend="ref")
    x, W1, W2, y, z = ref_op.gen_data()
    ref_op._ref_forward(x, W1, W2, y, z)
    z_ref = z.clone()

    ref_ms = bench_kernel(
        ref_op, args=(x, W1, W2, y, z),
        n_warmup=WARMUP, n_repeat=N_REPEAT, n_trials=N_TRIALS,
    )

    rows = []
    base_ms = None
    for backend in backends:
        try:
            op = PdlGemmOp(M, K, N, P, backend=backend)
            # Fresh scratch / output buffers for this backend.
            y_b = torch.empty_like(y)
            z_b = torch.empty_like(z)
            op(x, W1, W2, y_b, z_b)
            max_diff = (z_b - z_ref).abs().max().item()
            rel_l2 = (z_b - z_ref).norm().item() / max(z_ref.norm().item(), 1.0)
            passed = op.check(z_b, z_ref)
            ms = bench_kernel(
                op, args=(x, W1, W2, y_b, z_b),
                n_warmup=WARMUP, n_repeat=N_REPEAT, n_trials=N_TRIALS,
            )
            speedup_vs_ref = ref_ms / ms if ms > 0 else float("inf")
            if backend == "cuda":
                base_ms = ms
            speedup_vs_base = (base_ms / ms) if base_ms and ms > 0 else None
            rows.append([
                backend,
                f"{rel_l2:.2e}",
                "PASS" if passed else "FAIL",
                f"{ms * 1000:.2f}",
                f"{speedup_vs_ref:.2f}x",
                f"{speedup_vs_base:.2f}x" if speedup_vs_base is not None else "—",
            ])
        except Exception as e:
            print(f"  [{backend}] failed: {e!r}")
            continue

    rows.append(["ref", "—", "—", f"{ref_ms * 1000:.2f}",
                 "1.00x", f"{ref_ms / base_ms:.2f}x" if base_ms else "—"])

    print(tabulate(
        rows,
        headers=["backend", "rel_l2", "result", "latency(us)",
                 "vs ref", "vs cuda"],
        tablefmt="github",
    ))


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False, description="PDL GEMM benchmark")
    parser.add_argument("-h", action="store_true", dest="list_backends",
                        help="List available backends and exit")
    parser.add_argument("-k", "--backends", type=str, default=None,
                        help="Comma-separated list of backends to benchmark")
    parser.add_argument("-w", "--workloads", type=str, default=None,
                        help="Comma-separated workload labels to run")
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

    selected_labels = (
        set(s.strip() for s in args.workloads.split(",")) if args.workloads else None
    )

    for workload in WORKLOADS:
        *params, label = workload
        if selected_labels and label not in selected_labels:
            continue
        _run_workload(*params, label, backends)


if __name__ == "__main__":
    main()
