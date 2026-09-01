import os

os.environ.setdefault("TILELANG_CACHE_DIR", os.path.join(os.getcwd(), ".tilelang"))

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from tabulate import tabulate

from ezops import FusedQkNormRopeOp, list_backends
from ezops.ops.utils.bench import bench_kernel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks.hardware.gpu_specs import detect_profile

BACKENDS = list_backends("fused_qk_norm_rope")
print(BACKENDS)
WARMUP = 10
N_REPEAT = 50
N_TRIALS = 3

# qwen3 head layout: 16 query heads x 128, 8 kv heads (GQA group 2)
HEADS = dict(num_heads=16, head_dim=128, num_kv_heads=8)

WORKLOADS = [
    # (batch, max_seq_len, label)
    (1, 128, "qwen3-s128"),
    (1, 1024, "qwen3-s1k"),
    (1, 4096, "qwen3-s4k"),
    (1, 8192, "qwen3-s8k"),
    (1, 32768, "qwen3-s32k"),
    (2, 8192, "qwen3-b2-s8k"),
    (4, 8192, "qwen3-b4-s8k"),
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


def _run_workload(batch, max_seq_len, label, profile, backends):
    """Run all backends for a single workload and print two tables.

    The op mutates the KV cache's slot `position` in place (the decode step's
    KV write-back), but the write is idempotent for fixed inputs, so repeated
    benchmark iterations and the ref/backend runs share one data pack.
    """
    position = max_seq_len // 2
    print(f"\n{'=' * 60}")
    print(f"  FusedQkNormRope workload: {label}")
    print(f"  batch={batch}  heads=16q/8kv x 128")
    print(f"  KV cache: ({batch}, 8, {max_seq_len}, 128)  position={position}")
    print(f"{'=' * 60}\n")

    torch.manual_seed(42)
    ref_op = FusedQkNormRopeOp(batch, backend="ref", max_seq_len=max_seq_len, position=position, **HEADS)
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
            op = FusedQkNormRopeOp(batch, backend=backend, max_seq_len=max_seq_len, position=position, **HEADS)
            out = op(*data)
            max_diff = max(
                (a - e).abs().max().item() for a, e in zip(out, out_ref, strict=True)
            )
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
    parser = argparse.ArgumentParser(add_help=False, description="FusedQkNormRope kernel benchmark")
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

    for batch, max_seq_len, label in WORKLOADS:
        _run_workload(batch, max_seq_len, label, profile, backends)


if __name__ == "__main__":
    main()
