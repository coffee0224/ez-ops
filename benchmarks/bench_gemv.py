import torch
from tabulate import tabulate

from ezops import GemvOp, list_backends
from ezops.ops.utils.bench import bench_kernel

BACKENDS = list_backends("gemv")
M = 4096
N = 4096
WARMUP = 10
N_REPEAT = 50
N_TRIALS = 3


def _run_backend(backend, A, x, y_ref):
    op = GemvOp(M=M, N=N, backend=backend)
    _, _, y = op.gen_data()
    op(A, x, y)
    max_diff = (y - y_ref).abs().max().item()
    passed = op.check(y, y_ref)
    ms = bench_kernel(op, args=(A, x, y), n_warmup=WARMUP, n_repeat=N_REPEAT, n_trials=N_TRIALS)
    return max_diff, passed, ms


def main():
    ref_op = GemvOp(M=M, N=N, backend="triton")
    A, x, y_ref = ref_op.gen_data()
    ref_op._ref_forward(A, x, y_ref)

    ref_ms = bench_kernel(ref_op._ref_forward, args=(A, x, y_ref), n_warmup=WARMUP, n_repeat=N_REPEAT, n_trials=N_TRIALS)

    rows = []
    for backend in BACKENDS:
        try:
            max_diff, passed, ms = _run_backend(backend, A, x, y_ref)
            speedup = ref_ms / ms if ms > 0 else float("inf")
            rows.append([backend, f"{max_diff:.2e}", "PASS" if passed else "FAIL", f"{ms:.4f}", f"{speedup:.2f}x"])
        except Exception:
            continue

    rows.append(["ref", "—", "—", f"{ref_ms:.4f}", "1.00x"])
    print(tabulate(rows, headers=["backend", "max_diff", "result", "latency(ms)", "speedup"], tablefmt="github"))


if __name__ == "__main__":
    main()
