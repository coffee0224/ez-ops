#!/usr/bin/env python
"""Generate GPU roofline profile YAML.

Runs hardware microbenchmarks (HBM bandwidth + matmul peak FLOPS) and
generates a profile YAML file for roofline analysis.

Usage:
    python scripts/roofline_profile.py
    python scripts/roofline_profile.py --quick
    python scripts/roofline_profile.py --dtypes bf16,fp16 --output-dir profiles/
"""

import argparse
import sys
from pathlib import Path

# Add project root so we can import benchmarks as a package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.hardware.gpu_specs import detect_profile, get_specs


def _detect_gpu():
    import torch
    props = torch.cuda.get_device_properties(0)
    name = props.name
    compute_cap = f"{props.major}.{props.minor}"
    arch = f"sm_{props.major * 10 + props.minor}"
    profile = detect_profile(name)

    sm_count = props.multi_processor_count
    l2_cache_bytes = getattr(props, "L2_cache_size", None)

    # Shared memory per SM: try CUDA driver API via ctypes
    shared_mem_per_sm = None
    try:
        import ctypes
        cuda = ctypes.CDLL("libcuda.so.1")
        if cuda.cuInit(0) == 0:
            device = ctypes.c_int()
            if cuda.cuDeviceGet(ctypes.byref(device), 0) == 0:
                value = ctypes.c_int()
                # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81
                if cuda.cuDeviceGetAttribute(ctypes.byref(value), 81, device) == 0:
                    shared_mem_per_sm = value.value
    except Exception:
        pass

    return name, compute_cap, arch, profile, sm_count, l2_cache_bytes, shared_mem_per_sm


def _fmt_bytes(gbs):
    """Format GB/s → bytes/s in scientific notation (e.g. '4800e9')."""
    val = gbs * 1e9
    coeff = gbs
    if coeff == int(coeff):
        return f"{int(coeff)}e9"
    return f"{coeff:.1f}e9"


def _fmt_flops(tflops):
    """Format TFLOPS → FLOPS in scientific notation (e.g. '1979e12')."""
    coeff = tflops
    if coeff == int(coeff):
        return f"{int(coeff)}e12"
    return f"{coeff:.1f}e12"


def generate_yaml(gpu_name, compute_cap, sm_count, l2_cache_bytes,
                   shared_mem_per_sm, hbm_result, compute_results, specs):
    """Generate profile YAML string."""
    lines = [
        f"# GPU Profile: {gpu_name}",
        f"# Calibration: benchmarks/hardware/ microbenchmarks",
        f"#",
        f"# Only store theoretical + calibration here.",
        f"# effective = theoretical * calibration is computed by load_profile().",
        "",
        f"gpu: {gpu_name}",
        f"compute_capability: {compute_cap}",
        "",
        f"sm_count: {sm_count}",
    ]

    if l2_cache_bytes is not None:
        l2_mb = l2_cache_bytes / (1024 * 1024)
        lines.append(f"l2_cache_size: {l2_cache_bytes}          # {l2_mb:.0f} MB")
    if shared_mem_per_sm is not None:
        smem_kb = shared_mem_per_sm / 1024
        lines.append(f"shared_memory_per_sm: {shared_mem_per_sm}       # {smem_kb:.0f} KB")

    lines.append("")

    # HBM
    lines.append("hbm:")
    lines.append(f"  theoretical: {_fmt_bytes(hbm_result['theoretical_gbs'])}        # bytes/s, spec sheet")
    lines.append(f"  calibration: {hbm_result['calibration']:.3f}         # STREAM Triad")
    lines.append("")

    # Tensor core
    lines.append("tensor_core:")

    # Determine all dtypes to include
    spec_dtypes = set()
    if specs and "tensor_core_tflops" in specs:
        spec_dtypes = set(specs["tensor_core_tflops"].keys())

    measured_dtypes = set(compute_results.keys())
    all_dtypes = measured_dtypes | spec_dtypes

    # Canonical order
    dtype_order = ["bf16", "fp16", "tf32", "fp8", "fp32"]
    ordered = [d for d in dtype_order if d in all_dtypes]
    ordered.extend(sorted(all_dtypes - set(ordered)))

    # Find a measured calibration to reuse as placeholder for unmeasured dtypes
    placeholder_cal = None
    for dtype in ordered:
        r = compute_results.get(dtype)
        if r and r.get("calibration") is not None:
            placeholder_cal = r["calibration"]
            break

    for dtype in ordered:
        result = compute_results.get(dtype)
        theo_tflops = None
        if specs and "tensor_core_tflops" in specs:
            theo_tflops = specs["tensor_core_tflops"].get(dtype)

        is_measured = result and result.get("calibration") is not None
        cal = result["calibration"] if is_measured else placeholder_cal
        comment = "" if is_measured else "        # placeholder — not yet measured"

        lines.append(f"  {dtype}:")
        if theo_tflops:
            lines.append(f"    theoretical: {_fmt_flops(theo_tflops)}     # FLOPS, spec sheet (dense)")
        if cal is not None:
            lines.append(f"    calibration: {cal:.2f}{comment}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate GPU roofline profile YAML")
    parser.add_argument("--output-dir", default="assets", help="Output directory (default: assets)")
    parser.add_argument("--dtypes", default="bf16,fp16,tf32,fp8",
                        help="Dtypes to benchmark, comma-separated (default: bf16,fp16,tf32,fp8)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer iterations, smaller sweep")
    args = parser.parse_args()

    # Import here so --help works without a GPU
    from benchmarks.hardware.compute.matmul_peak import run as run_matmul
    from benchmarks.hardware.memory.hbm_bandwidth import run as run_hbm

    # Detect GPU
    gpu_name, compute_cap, arch, profile, sm_count, l2_cache_bytes, shared_mem_per_sm = _detect_gpu()
    specs = get_specs(profile) if profile else None

    print(f"GPU: {gpu_name}")
    print(f"Profile: {profile or 'unknown'}")
    print(f"Compute capability: {compute_cap}")
    print(f"SM count: {sm_count}")
    if l2_cache_bytes is not None:
        print(f"L2 cache: {l2_cache_bytes / (1024 * 1024):.0f} MB")
    if shared_mem_per_sm is not None:
        print(f"Shared memory per SM: {shared_mem_per_sm / 1024:.0f} KB")
    print()

    if not profile:
        print("WARNING: GPU not in specs database. Theoretical values will be unavailable.")
        print("Calibration cannot be computed without theoretical peaks.")
        print()

    # Resolve theoretical HBM BW
    hbm_theo = specs["hbm_bw_gb"] if specs else None
    dtypes = [d.strip() for d in args.dtypes.split(",")]

    # ── Phase 1: HBM Bandwidth ────────────────────────────────────────────
    print("=" * 60)
    print("Phase 1: HBM Bandwidth (STREAM Triad)")
    print("=" * 60)
    hbm_result = run_hbm(
        profile=profile,
        arch=arch,
        theoretical_bw=hbm_theo,
        size_mb=512 if args.quick else 2048,
    )
    print(f"\n  Measured: {hbm_result['measured_gbs']:.2f} GB/s")
    print(f"  Theoretical: {hbm_result['theoretical_gbs']:.1f} GB/s")
    print(f"  Calibration: {hbm_result['calibration']:.4f}")
    print()

    # ── Phase 2: Matmul Peak FLOPS ────────────────────────────────────────
    print("=" * 60)
    print(f"Phase 2: Matmul Peak FLOPS ({', '.join(dtypes)})")
    print("=" * 60)

    compute_results = {}
    for dtype in dtypes:
        print(f"\n--- {dtype} ---")
        try:
            result = run_matmul(
                dtype=dtype,
                warmup=5 if args.quick else 30,
                n_iter=20 if args.quick else 100,
                m_step=1024 if args.quick else 512,
                gpu_warmup_secs=3 if args.quick else 5,
            )
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        compute_results[dtype] = result
        print(f"\n  Best: M={result['best_shape'][0]}, N={result['best_shape'][1]}, "
              f"K={result['best_shape'][2]}  →  {result['best_tflops']:.1f} TFLOPS")
        if result.get("theoretical_tflops"):
            print(f"  Theoretical: {result['theoretical_tflops']:.0f} TFLOPS  →  "
                  f"{result['calibration']*100:.1f}% utilization")

    # ── Generate YAML ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Generating profile YAML")
    print("=" * 60)

    yaml_str = generate_yaml(gpu_name, compute_cap, sm_count, l2_cache_bytes,
                             shared_mem_per_sm, hbm_result, compute_results, specs)

    filename = f"{profile or gpu_name.lower().replace(' ', '_')}.yaml"
    output_path = Path(args.output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_str)

    print(f"\nSaved to: {output_path}")
    print()
    print(yaml_str)


if __name__ == "__main__":
    main()
