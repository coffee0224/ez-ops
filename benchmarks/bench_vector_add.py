import torch

from ezops import VectorAddOp, list_backends
from ezops.ops.utils.bench import bench_kernel

BACKENDS = list_backends("vector_add")
N = 1 << 20
WARMUP = 10
N_REPEAT = 50
N_TRIALS = 3


def bench_correctness():
    ref_op = VectorAddOp(n=N, backend="triton")
    A, B, C_ref = ref_op.gen_data()
    ref_op._ref_forward(A, B, C_ref)

    print(f"{'backend':<12} {'max_diff':>12} {'result':>10}")
    print("-" * 36)

    for backend in BACKENDS:
        try:
            op = VectorAddOp(n=N, backend=backend)
        except Exception as e:
            print(f"{backend:<12} {'—':>12} {'ERROR':>10}  {e}")
            continue

        _, _, C = op.gen_data()
        op(A, B, C)

        max_diff = (C - C_ref).abs().max().item()
        passed = op.check(C, C_ref)
        print(f"{backend:<12} {max_diff:>12.2e} {'PASS' if passed else 'FAIL':>10}")


def bench_latency():
    ref_op = VectorAddOp(n=N, backend="triton")
    A, B, C = ref_op.gen_data()

    print(f"{'backend':<12} {'latency_ms':>12}")
    print("-" * 26)

    for backend in BACKENDS:
        try:
            op = VectorAddOp(n=N, backend=backend)
        except Exception as e:
            print(f"{backend:<12} {'ERROR':>12}  {e}")
            continue

        ms = bench_kernel(op, args=(A, B, C), n_warmup=WARMUP, n_repeat=N_REPEAT, n_trials=N_TRIALS)
        print(f"{backend:<12} {ms:>12.4f}")


if __name__ == "__main__":
    bench_correctness()
    print()
    bench_latency()
