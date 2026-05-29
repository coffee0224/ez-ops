import subprocess
import sys
from pathlib import Path

import torch
import yaml
from tabulate import tabulate

from ezops import AttnDecodeOp, list_backends
from ezops.ops.utils.bench import bench_kernel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks.hardware.gpu_specs import detect_profile

BACKENDS = list_backends("attn_decode")
print(BACKENDS)
WARMUP = 10
N_REPEAT = 50
N_TRIALS = 3

WORKLOADS = [
    # (batch, num_heads, seq_len, head_dim, label)
    (1, 32, 1024, 128, "llama-7b-s1k"),
    (1, 32, 4096, 128, "llama-7b-s4k"),
    (1, 32, 8192, 128, "llama-7b-s8k"),
    (1, 32, 16384, 128, "llama-7b-s16k"),
    (1, 32, 32768, 128, "llama-7b-s16k"),
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


def _run_workload(batch, num_heads, seq_len, head_dim, label, profile):
    """Run all backends for a single workload and print two tables."""
    print(f"\n{'=' * 60}")
    print(f"  AttnDecode workload: {label}")
    print(f"  Q: ({batch}, {num_heads}, 1, {head_dim})  K/V: ({batch}, {num_heads}, {seq_len}, {head_dim})")
    print(f"{'=' * 60}\n")

    torch.manual_seed(42)
    ref_op = AttnDecodeOp(batch, num_heads, seq_len, head_dim, backend="ref")
    Q, K, V = ref_op.gen_data()
    out_ref = ref_op._ref_forward(Q, K, V)
    roofline = ref_op.get_roofline()

    ref_ms = bench_kernel(
        ref_op._ref_forward,
        args=(Q, K, V),
        n_warmup=WARMUP,
        n_repeat=N_REPEAT,
        n_trials=N_TRIALS,
    )

    sol = _compute_sol(roofline, profile, Q.nbytes + K.nbytes + V.nbytes) if profile else None

    rows = []
    for backend in BACKENDS:
        try:
            op = AttnDecodeOp(batch, num_heads, seq_len, head_dim, backend=backend)
            out = op(Q, K, V)
            max_diff = (out - out_ref).abs().max().item()
            passed = op.check(out, out_ref)
            ms = bench_kernel(op, args=(Q, K, V), n_warmup=WARMUP, n_repeat=N_REPEAT, n_trials=N_TRIALS)
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


def main():
    profile_name = _detect_gpu_profile()
    profile = _load_profile(profile_name) if profile_name else None

    for batch, num_heads, seq_len, head_dim, label in WORKLOADS:
        _run_workload(batch, num_heads, seq_len, head_dim, label, profile)


if __name__ == "__main__":
    main()
