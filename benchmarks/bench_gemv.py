import subprocess
import sys
from pathlib import Path

import torch
import yaml
from tabulate import tabulate

from ezops import GemvOp, list_backends
from ezops.ops.utils.bench import bench_kernel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks.hardware.gpu_specs import detect_profile

BACKENDS = list_backends("gemv")
N = 4096
K = 4096
WARMUP = 10
N_REPEAT = 50
N_TRIALS = 3


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


def _run_backend(backend, A, B, C_ref):
    op = GemvOp(N=N, K=K, backend=backend)
    # Adapt shapes for kernel buffers: A(K,), B(N,K), C(N,)
    A_ad = A.squeeze(0)
    B_ad = B.T.contiguous()
    C_ad = torch.empty(N, device="cuda", dtype=torch.bfloat16)
    op(A_ad, B_ad, C_ad)
    C = C_ad.unsqueeze(0)
    max_diff = (C - C_ref).abs().max().item()
    passed = op.check(C, C_ref)
    ms = bench_kernel(op, args=(A_ad, B_ad, C_ad), n_warmup=WARMUP, n_repeat=N_REPEAT, n_trials=N_TRIALS)
    return max_diff, passed, ms


def main():
    torch.manual_seed(42)
    ref_op = GemvOp(N=N, K=K, backend="triton")
    A, B, C_ref = ref_op.gen_data()
    ref_op._ref_forward(A, B, C_ref)
    roofline = ref_op.get_roofline()

    ref_ms = bench_kernel(
        ref_op._ref_forward, args=(A, B, C_ref),
        n_warmup=WARMUP, n_repeat=N_REPEAT, n_trials=N_TRIALS,
    )

    # SOL analysis
    profile_name = _detect_gpu_profile()
    profile = _load_profile(profile_name) if profile_name else None
    sol = _compute_sol(roofline, profile, A.nbytes + B.nbytes) if profile else None

    rows = []
    for backend in BACKENDS:
        try:
            max_diff, passed, ms = _run_backend(backend, A, B, C_ref)
            speedup = ref_ms / ms if ms > 0 else float("inf")
            sol_score = sol["theo_min_s"] / (ms / 1000) if sol else None
            rows.append([
                backend, f"{max_diff:.2e}", "PASS" if passed else "FAIL",
                f"{ms:.4f}", f"{speedup:.2f}x",
                f"{sol_score:.1f}x" if sol_score is not None else "—",
            ])
        except Exception:
            continue

    ref_sol = sol["theo_min_s"] / (ref_ms / 1000) if sol else None
    rows.append([
        "ref", "—", "—", f"{ref_ms:.4f}", "1.00x",
        f"{ref_sol:.1f}x" if ref_sol is not None else "—",
    ])

    print(tabulate(
        rows,
        headers=["backend", "max_diff", "result", "latency(ms)", "speedup", "sol-score"],
        tablefmt="github",
    ))

    if sol:
        print()
        sol_rows = [
            ["compute time (tf32 peak)", f"{sol['compute_us']:.3f} µs"],
            ["mem time (fused)", f"{sol['mem_fused_us']:.3f} µs"],
            ["mem time (unfused)", f"{sol['mem_unfused_us']:.3f} µs"],
            ["theoretical min", f"{sol['theo_min_us']:.3f} µs"],
        ]
        print(tabulate(sol_rows, headers=["SOL metric", "value"], tablefmt="github"))


if __name__ == "__main__":
    main()
