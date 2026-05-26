import torch

from ezops import VectorAddOp, list_backends

BACKENDS = list_backends("vector_add")
N = 1 << 20
WARMUP = 10
ITERS = 100


def bench_correctness():
    ref_op = VectorAddOp(n=N, backend="triton")
    A, B, C_ref = ref_op.gen_data()
    ref_op._ref_forward(A, B, C_ref)

    print(f"{'backend':<12} {'max_diff':>12} {'result':>10}  {'latency':>10}")
    print("-" * 48)

    for backend in BACKENDS:
        try:
            op = VectorAddOp(n=N, backend=backend)
        except Exception as e:
            print(f"{backend:<12} {'—':>12} {'ERROR':>10}  {e}")
            continue

        _, _, C = op.gen_data()
        for _ in range(WARMUP):
            op(A, B, C)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            op(A, B, C)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / ITERS

        max_diff = (C - C_ref).abs().max().item()
        passed = torch.allclose(C, C_ref, atol=1e-6, rtol=1e-5)
        print(f"{backend:<12} {max_diff:>12.2e} {'PASS' if passed else 'FAIL':>10}  {ms:.4f} ms")


if __name__ == "__main__":
    bench_correctness()
