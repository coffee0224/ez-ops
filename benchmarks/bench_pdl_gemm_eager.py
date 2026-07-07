"""Eager PDL comparison: same kernels, but timed WITHOUT CUDA Graph capture
so we measure the real end-to-end latency including CPU dispatch.

For tiny kernels, the CPU dispatch overhead is a significant fraction of
total latency. PDL overlaps the second kernel's CPU dispatch + GPU prolog
with the first kernel's GPU execution, so the eager case is where PDL
shines most visibly.
"""

import statistics
import sys
from pathlib import Path

import torch
from tabulate import tabulate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ezops import PdlGemmOp

WORKLOADS = [
    (64,  128,  128,  128, "tiny-mlp"),
    (64,  512,  512,  512, "small-mlp"),
    (128, 1024, 1024, 1024, "medium-mlp"),
    (256, 2048, 2048, 2048, "large-mlp"),
]

WARMUP = 30
N_REPEAT = 200
N_TRIALS = 5


def bench_eager(fn, args, n_warmup, n_repeat, n_trials):
    """Time fn(*args) on the GPU using CUDA events, no graph capture.

    Returns median latency across trials. Within each trial we take the
    median across repeats (not the mean) so occasional slow cudaLaunchKernelEx
    dispatches don't drag the number up."""
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()

    trial_medians = []
    for _ in range(n_trials):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
        for i in range(n_repeat):
            starts[i].record()
            fn(*args)
            ends[i].record()
        torch.cuda.synchronize()
        times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends, strict=True))
        trial_medians.append(times[len(times) // 2])
    trial_medians.sort()
    return trial_medians[len(trial_medians) // 2]


def run_workload(M, K, N, P, label):
    print(f"\n{'=' * 70}")
    print(f"  Eager PDL GEMM: {label}  (M={M}, K={K}, N={N}, P={P})")
    print(f"{'=' * 70}\n")

    torch.manual_seed(42)
    ref_op = PdlGemmOp(M, K, N, P, backend="ref")
    x, W1, W2, y, z = ref_op.gen_data()
    ref_op._ref_forward(x, W1, W2, y, z)
    z_ref = z.clone()

    rows = []
    base_ms = None
    for backend in ["ref", "cuda", "cuda_pdl"]:
        if backend == "ref":
            op = ref_op
            fn = ref_op._ref_forward
        else:
            op = PdlGemmOp(M, K, N, P, backend=backend)
            fn = op
        y_b = torch.empty_like(y)
        z_b = torch.empty_like(z)
        fn(x, W1, W2, y_b, z_b)
        max_diff = (z_b - z_ref).abs().max().item()
        passed = op.check(z_b, z_ref)

        ms = bench_eager(fn, (x, W1, W2, y_b, z_b), WARMUP, N_REPEAT, N_TRIALS)
        if backend == "cuda":
            base_ms = ms
        vs_base = (base_ms / ms) if base_ms and backend != "ref" else None

        rows.append([
            backend,
            f"{max_diff:.2e}",
            "PASS" if passed else "FAIL",
            f"{ms * 1e3:.2f}",
            f"{vs_base:.2f}x" if vs_base is not None else "—",
        ])

    print(tabulate(
        rows,
        headers=["backend", "max_diff", "result", "latency(us)", "vs cuda"],
        tablefmt="github",
    ))


def main():
    for M, K, N, P, label in WORKLOADS:
        run_workload(M, K, N, P, label)


if __name__ == "__main__":
    main()
