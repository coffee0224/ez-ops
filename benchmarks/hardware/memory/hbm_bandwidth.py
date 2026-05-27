"""HBM Bandwidth Benchmark — Python wrapper for hbm_saturation.cu.

Compiles and runs the CUDA microbenchmark, parses output, and prints
the measured vs theoretical HBM bandwidth.

Calibration is derived from the STREAM Triad kernel (a = b + s*c, 2 reads +
1 write).  Triad is the industry standard for roofline bandwidth calibration:

    McCalpin, J.D., 1995. "Memory Bandwidth and Machine Balance in Current
    High Performance Computers." IEEE TCCA Newsletter.
    https://www.cs.virginia.edu/stream/

    Williams, S., Waterman, A. & Patterson, D., 2009. "Roofline: An Insightful
    Visual Performance Model for Multicore Architectures." CACM 52(4).

Usage:
    python benchmarks/hardware/memory/hbm_bandwidth.py [--size-mb 2048]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from ..gpu_specs import detect_profile, get_specs

_CU_SRC = Path(__file__).parent / "hbm_saturation.cu"


def _detect_gpu():
    """Auto-detect GPU name and compute capability from nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return None, None, None
    if r.returncode != 0:
        return None, None, None

    line = r.stdout.strip().split("\n")[0]
    name, cap = [s.strip() for s in line.split(",")]
    major, minor = cap.split(".")
    arch = f"sm_{int(major) * 10 + int(minor)}"
    profile = detect_profile(name)
    return name, arch, profile


def _compile(cu_path, binary_path, arch):
    """Compile the CUDA source. Raises on failure."""
    cmd = [
        "nvcc", "-O3", f"-arch={arch}",
        "-Wno-deprecated-gpu-targets",
        "-o", str(binary_path), str(cu_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc compilation failed:\n{result.stderr}")


def _run_binary(binary_path, size_mb, theo_peak_gbs):
    """Run the benchmark binary and return stdout lines."""
    cmd = [str(binary_path), str(size_mb), str(theo_peak_gbs)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed:\n{result.stderr}")
    return result.stdout.strip().splitlines()


def _parse_triad_peak(lines):
    """Extract the best Triad bandwidth (GB/s) from CSV output."""
    best_gbs = 0.0
    for line in lines:
        if not line.startswith("triad,"):
            continue
        parts = line.split(",")
        if len(parts) >= 6:
            try:
                gbs = float(parts[5])
                best_gbs = max(best_gbs, gbs)
            except ValueError:
                continue
    return best_gbs


def run(profile=None, arch=None, theoretical_bw=None, size_mb=2048, quiet=False):
    """Run HBM bandwidth benchmark.

    Args:
        profile: GPU profile key (auto-detect if None).
        arch: CUDA arch for nvcc, e.g. "sm_90" (auto-detect if None).
        theoretical_bw: Theoretical HBM BW in GB/s (auto-detect from profile if None).
        size_mb: Working set size in MB.
        quiet: Suppress informational output.

    Returns:
        dict with: gpu_name, arch, profile, measured_gbs, theoretical_gbs, calibration.

    Raises:
        RuntimeError: If compilation or benchmark execution fails.
        ValueError: If theoretical BW cannot be determined.
    """
    gpu_name, gpu_arch, gpu_profile = _detect_gpu()

    profile = profile or gpu_profile
    arch = arch or gpu_arch

    if not arch:
        raise ValueError("Cannot detect CUDA arch. Pass arch explicitly (e.g. sm_90).")

    # Resolve theoretical BW
    if theoretical_bw is not None:
        theo_peak_gbs = theoretical_bw
    elif profile:
        specs = get_specs(profile)
        if specs and "hbm_bw_gb" in specs:
            theo_peak_gbs = specs["hbm_bw_gb"]
        else:
            raise ValueError(f"No HBM BW spec for profile '{profile}'. Pass theoretical_bw.")
    else:
        raise ValueError("Cannot determine theoretical BW. Pass profile or theoretical_bw.")

    if not quiet:
        print(f"GPU: {gpu_name or 'unknown'}")
        print(f"Profile: {profile}")
        print(f"Arch: {arch}")
        print(f"Theoretical HBM BW: {theo_peak_gbs:.1f} GB/s")
        print(f"Working set: {size_mb} MB")
        print()
        print("Compiling hbm_saturation.cu ...")

    with tempfile.TemporaryDirectory() as tmpdir:
        binary = Path(tmpdir) / "hbm_saturation"
        _compile(_CU_SRC, binary, arch=arch)

        if not quiet:
            print("Running benchmark (5 runs x 200 reps, this may take a few minutes) ...\n")

        lines = _run_binary(binary, size_mb, theo_peak_gbs)

    if not quiet:
        for line in lines:
            print(line)

    measured_gbs = _parse_triad_peak(lines)
    calibration = measured_gbs / theo_peak_gbs if theo_peak_gbs > 0 else 0.0

    if not quiet and measured_gbs > 0:
        print(f"\n{'='*60}")
        print(f"Measured peak (triad vec4): {measured_gbs:.2f} GB/s")
        print(f"Theoretical:               {theo_peak_gbs:.1f} GB/s")
        print(f"Calibration:               {calibration:.4f}")
        print(f"{'='*60}")

    return {
        "gpu_name": gpu_name,
        "arch": arch,
        "profile": profile,
        "measured_gbs": measured_gbs,
        "theoretical_gbs": theo_peak_gbs,
        "calibration": calibration,
    }


def main():
    gpu_name, gpu_arch, gpu_profile = _detect_gpu()

    parser = argparse.ArgumentParser(description="HBM bandwidth microbenchmark")
    parser.add_argument("--profile", default=gpu_profile,
                        help=f"GPU profile name (auto: {gpu_profile})")
    parser.add_argument("--arch", default=gpu_arch,
                        help=f"CUDA arch for nvcc (auto: {gpu_arch})")
    parser.add_argument("--theoretical-bw", type=float, default=None,
                        help="Theoretical peak HBM BW in GB/s (overrides profile lookup)")
    parser.add_argument("--size-mb", type=int, default=2048,
                        help="Working set size in MB")
    args = parser.parse_args()

    try:
        result = run(
            profile=args.profile,
            arch=args.arch,
            theoretical_bw=args.theoretical_bw,
            size_mb=args.size_mb,
        )
    except (RuntimeError, ValueError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
