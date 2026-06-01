import argparse
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from tabulate import tabulate

from ezops import VectorAddOp, list_backends
from ezops.ops.utils.bench import bench_kernel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks.hardware.gpu_specs import detect_profile

BACKENDS = list_backends("vector_add")
print(BACKENDS)
WARMUP = 10
N_REPEAT = 50
N_TRIALS = 3

WORKLOADS = [
    # (n, label)
    (1 << 10, "1K"),
    (1 << 20, "1M"),
    (1 << 24, "16M"),
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


def _run_workload(n, label, profile, backends):
    """Run all backends for a single workload and print two tables."""
    print(f"\n{'=' * 60}")
    print(f"  VectorAdd workload: {label}  (n={n})")
    print(f"  A: ({n},)  B: ({n},)  C: ({n},)")
    print(f"{'=' * 60}\n")

    torch.manual_seed(42)
    ref_op = VectorAddOp(n=n, backend="ref")
    A, B, C_ref = ref_op.gen_data()
    ref_op._ref_forward(A, B, C_ref)
    roofline = ref_op.get_roofline()

    ref_ms = bench_kernel(
        ref_op._ref_forward, args=(A, B, C_ref),
        n_warmup=WARMUP, n_repeat=N_REPEAT, n_trials=N_TRIALS,
    )

    sol = _compute_sol(roofline, profile, A.nbytes + B.nbytes) if profile else None

    rows = []
    for backend in backends:
        try:
            op = VectorAddOp(n=n, backend=backend)
            _, _, C = op.gen_data()
            op(A, B, C)
            max_diff = (C - C_ref).abs().max().item()
            passed = op.check(C, C_ref)
            ms = bench_kernel(op, args=(A, B, C), n_warmup=WARMUP, n_repeat=N_REPEAT, n_trials=N_TRIALS)
            speedup = ref_ms / ms if ms > 0 else float("inf")
            sol_score = sol["theo_min_s"] / (ms / 1000) if sol else None
            rows.append([
                backend, f"{max_diff:.2e}", "PASS" if passed else "FAIL",
                f"{ms:.4f}", f"{speedup:.2f}x",
                f"{sol_score:.1f}x" if sol_score is not None else "—",
            ])
        except Exception as e:
            print(e)
            continue

    ref_sol = sol["theo_min_s"] / (ref_ms / 1000) if sol else None
    rows.append([
        "ref", "—", "—", f"{ref_ms:.4f}", "1.00x",
        f"{ref_sol:.1f}x" if ref_sol is not None else "—",
    ])

    # Table 1: Performance
    print(tabulate(
        rows,
        headers=["backend", "max_diff", "result", "latency(ms)", "speedup", "sol-score"],
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
    parser = argparse.ArgumentParser(add_help=False, description="VectorAdd kernel benchmark")
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

    for n, label in WORKLOADS:
        _run_workload(n, label, profile, backends)


if __name__ == "__main__":
    main()
