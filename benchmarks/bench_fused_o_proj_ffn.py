import os

os.environ.setdefault("TILELANG_CACHE_DIR", os.path.join(os.getcwd(), ".tilelang"))

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from tabulate import tabulate

from ezops import FusedOProjFfnOp, list_backends
from ezops.ops.utils.bench import bench_kernel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks.hardware.gpu_specs import detect_profile

BACKENDS = list_backends("fused_o_proj_ffn")
print(BACKENDS)
WARMUP = 10
N_REPEAT = 50
N_TRIALS = 3

# qwen3 dense decoder layer shapes: hidden, intermediate, 16 heads x 128
# (2048 attention lanes; wider than the 0.6B hidden)
SHAPES = {
    "qwen3-0.6b": dict(hidden_size=1024, intermediate_size=3072, num_heads=16, head_dim=128),
    "qwen3-vl-2b": dict(hidden_size=2048, intermediate_size=6144, num_heads=16, head_dim=128),
}

WORKLOADS = [
    # (batch, shape key, label) — decode-stage op, benchmarked at batch 1
    (1, "qwen3-0.6b", "qwen3-0.6b-b1"),
    (1, "qwen3-vl-2b", "qwen3-vl-2b-b1"),
]


def _detect_gpu_profile():
    """Detect current GPU via nvidia-smi and return profile key."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
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


def _run_workload(batch, shape_key, label, profile, backends):
    """Run all backends for a single workload and print two tables."""
    shape = SHAPES[shape_key]
    print(f"\n{'=' * 60}")
    print(f"  FusedOProjFfn workload: {label}")
    print(f"  batch={batch}  hidden={shape['hidden_size']}  intermediate={shape['intermediate_size']}")
    print(f"  o_proj input: 16 heads x 128 = 2048 lanes")
    print(f"{'=' * 60}\n")

    torch.manual_seed(42)
    ref_op = FusedOProjFfnOp(batch, backend="ref", **shape)
    data = ref_op.gen_data()
    out_ref = ref_op._ref_forward(*data)
    roofline = ref_op.get_roofline()

    ref_ms = bench_kernel(
        ref_op._ref_forward,
        args=data,
        n_warmup=WARMUP,
        n_repeat=N_REPEAT,
        n_trials=N_TRIALS,
    )

    input_bytes = sum(t.nbytes for t in data)
    sol = _compute_sol(roofline, profile, input_bytes) if profile else None

    rows = []
    for backend in backends:
        try:
            op = FusedOProjFfnOp(batch, backend=backend, **shape)
            out = op(*data)
            max_diff = (out - out_ref).abs().max().item()
            passed = op.check(out, out_ref)
            ms = bench_kernel(op, args=data, n_warmup=WARMUP, n_repeat=N_REPEAT, n_trials=N_TRIALS)
            speedup = ref_ms / ms if ms > 0 else float("inf")
            sol_score = sol["theo_min_s"] / (ms / 1000) if sol else None
            rows.append(
                [
                    backend,
                    f"{max_diff:.2e}",
                    "PASS" if passed else "FAIL",
                    f"{ms:.4f}",
                    f"{speedup:.2f}x",
                    f"{sol_score:.1f}x" if sol_score is not None else "—",
                ]
            )
        except Exception as e:
            print(e)
            continue

    ref_sol = sol["theo_min_s"] / (ref_ms / 1000) if sol else None
    rows.append(
        [
            "ref",
            "—",
            "—",
            f"{ref_ms:.4f}",
            "1.00x",
            f"{ref_sol:.1f}x" if ref_sol is not None else "—",
        ]
    )

    # Table 1: Performance
    print(
        tabulate(
            rows,
            headers=["backend", "max_diff", "result", "latency(ms)", "speedup", "sol-score"],
            tablefmt="github",
        )
    )

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
    parser = argparse.ArgumentParser(add_help=False, description="FusedOProjFfn kernel benchmark")
    parser.add_argument("-h", action="store_true", dest="list_backends", help="List available backends and exit")
    parser.add_argument(
        "-k",
        "--backends",
        type=str,
        default=None,
        help="Comma-separated list of backends to benchmark (ref is always included)",
    )
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

    for batch, shape_key, label in WORKLOADS:
        _run_workload(batch, shape_key, label, profile, backends)


if __name__ == "__main__":
    main()
