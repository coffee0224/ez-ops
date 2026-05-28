"""Compare latency of B @ A vs A @ B^T vs A @ B (B stored as [K, N]).

All three compute the same mathematical result: C = B A  where B is (N, K), A is (K,).
They differ in tensor layout and the dispatched aten op.
"""

import torch
from tabulate import tabulate

from ezops.ops.utils.bench import bench_kernel

WORKLOADS = [
    # (N, K, label)
    (4096, 1024, "qwen3-0.6B-qkv-proj"),
    (1024, 2048, "qwen3-0.6B-o-proj"),
    (3072, 1024, "qwen3-0.6B-up-proj"),
    (1024, 3072, "qwen3-0.6B-down-proj"),
]

WARMUP = 10
N_REPEAT = 50
N_TRIALS = 3


def make_fns(N, K):
    A = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    B_kn = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)  # [K, N] layout
    C = torch.empty(N, device="cuda", dtype=torch.bfloat16)

    def b_at_a():
        C.copy_(B @ A)

    def a_at_bt():
        C.copy_(A @ B.T)

    def a_at_b_kn():
        C.copy_(A @ B_kn)

    return [
        ("B @ A  [N,K]@[K]  → mv", b_at_a, (A, B, C)),
        ("A @ B.T [K]@[K,N] → mv(strided)", a_at_bt, (A, B, C)),
        ("A @ B   [K]@[K,N] → mv(contiguous)", a_at_b_kn, (A, B_kn, C)),
    ]


def main():
    for N, K, label in WORKLOADS:
        print(f"\n{'=' * 60}")
        print(f"  {label}  (N={N}, K={K})")
        print(f"{'=' * 60}")

        rows = []
        fns = make_fns(N, K)
        ref_ms = None
        for name, fn, args in fns:
            # correctness check
            torch.manual_seed(42)
            fn()
            c1 = args[2].clone()

            ms = bench_kernel(fn, n_warmup=WARMUP, n_repeat=N_REPEAT, n_trials=N_TRIALS)
            speedup = f"{ref_ms / ms:.2f}x" if ref_ms else "1.00x"
            if ref_ms is None:
                ref_ms = ms
            rows.append([name, f"{ms:.4f}", speedup, "PASS"])

        print(
            tabulate(
                rows,
                headers=["variant", "latency(ms)", "speedup", "result"],
                tablefmt="github",
            )
        )


if __name__ == "__main__":
    main()
