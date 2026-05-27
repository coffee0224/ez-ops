"""Matmul Peak FLOPS Benchmark.

Sweeps matmul (M, N, K) shapes to find the Maximum Achievable Matmul FLOPS (MAMF)
on the current GPU. Uses cuBLAS via PyTorch with proper benchmarking hygiene:

  1. L2 cache flush via large dummy buffer before each measurement
  2. Destination matrix re-randomization to prevent cuBLAS fast paths
  3. CUDA event timing (wall-clock not reliable for async GPU ops)
  4. GPU warmup period to reach steady-state clock

Methodology derived from:
    mamf-finder.py in stas00/ml-engineering
    https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py

Usage:
    python benchmarks/hardware/compute/matmul_peak.py
    python benchmarks/hardware/compute/matmul_peak.py --dtype fp16 --top-k 10
"""

import argparse
import time

import torch
from tabulate import tabulate

from ..gpu_specs import detect_profile, get_specs

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "tf32": torch.float32,  # tf32 is a compute mode, not a storage dtype
}


def _detect_gpu():
    props = torch.cuda.get_device_properties(0)
    return props.name, props.total_memory


def _max_m_for_memory(n, k, dtype, total_mem_bytes, safety=0.8):
    elem_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    max_usable = total_mem_bytes * safety
    b_bytes = k * n * elem_bytes  # B(K,N) is fixed
    per_m = (k + n) * elem_bytes  # one row of A(M,K) + one row of C(M,N)
    available = max_usable - b_bytes
    if available <= 0 or per_m <= 0:
        return 256
    return max(256, int(available / per_m))


def _flush_l2(cache_buf):
    cache_buf.fill_(0x5A5A5A5A)


def _benchmark_shape(M, N, K, dtype, n_warmup, n_iter, cache_buf, use_tf32=False):
    """Run matmul benchmark for a single shape, return (mean, median, max) TFLOPS."""
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    c = torch.randn(M, N, device="cuda", dtype=dtype)

    # Warmup
    for _ in range(n_warmup):
        torch.mm(a, b, out=c)

    torch.cuda.synchronize()

    flops = 2.0 * M * N * K
    times = []

    for _ in range(n_iter):
        _flush_l2(cache_buf)
        c.normal_()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(a, b, out=c)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end) / 1000.0)  # ms → s

    median_s = sorted(times)[len(times) // 2]
    tflops = flops / median_s / 1e12
    mean_s = sum(times) / len(times)
    tflops_mean = flops / mean_s / 1e12
    tflops_max = flops / min(times) / 1e12

    return tflops_mean, tflops, tflops_max


def _phase1_shapes(max_m, step=512, n=4096, k=4096):
    """Coarse sweep: M from 256 to max_m in large steps."""
    shapes = []
    m = 256
    while m <= max_m:
        shapes.append((m, n, k))
        m += step
    return shapes


def _phase2_shapes(top_ms, n=4096, k=4096, radius=512, step=128):
    """Fine sweep around the best M values from phase 1."""
    shapes = []
    seen = set()
    for m in top_ms:
        lo = max(256, m - radius)
        hi = m + radius
        cur = lo
        while cur <= hi:
            if cur not in seen:
                shapes.append((cur, n, k))
                seen.add(cur)
            cur += step
    return shapes


def run(dtype="bf16", n=4096, k=4096, m_min=256, m_max=None, m_step=512,
        phase2_radius=512, phase2_step=128, warmup=30, n_iter=100, top_k=5,
        cache_mb=256, gpu_warmup_secs=5, quiet=False):
    """Run matmul peak FLOPS benchmark.

    Args:
        dtype: "bf16", "fp16", "fp32", or "tf32".
        n: N dimension.
        k: K dimension.
        m_min: Minimum M dimension.
        m_max: Maximum M dimension (auto from GPU memory if None).
        m_step: Phase 1 M step.
        phase2_radius: Phase 2 search radius around top M values.
        phase2_step: Phase 2 M step.
        warmup: Warmup iterations per shape.
        n_iter: Measurement iterations per shape.
        top_k: Number of top shapes to refine in phase 2.
        cache_mb: L2 flush buffer size in MB.
        gpu_warmup_secs: GPU steady-state warmup in seconds.
        quiet: Suppress informational output.

    Returns:
        dict with: gpu_name, dtype, best_tflops, best_shape, theoretical_tflops,
                   calibration, all_results.

    Raises:
        ValueError: If dtype is not supported.
    """
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype}'. Choose from: {sorted(_DTYPE_MAP)}")

    use_tf32 = dtype == "tf32"
    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    torch_dtype = _DTYPE_MAP[dtype]

    gpu_name, total_mem = _detect_gpu()
    profile = detect_profile(gpu_name)
    specs = get_specs(profile) if profile else None
    theoretical_tflops = None
    if specs and "tensor_core_tflops" in specs:
        theoretical_tflops = specs["tensor_core_tflops"].get(dtype)

    if m_max is None:
        m_max = _max_m_for_memory(n, k, torch_dtype, total_mem)
        m_max = min(m_max, 16384)

    if not quiet:
        print(f"GPU: {gpu_name}")
        print(f"Memory: {total_mem / 1e9:.1f} GB")
        print(f"Dtype: {dtype}")
        print(f"Sweep: M=[{m_min}..{m_max}], N={n}, K={k}")

    # L2 flush buffer
    cache_buf = torch.empty(cache_mb * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)

    # GPU warmup: run heavy matmul to reach steady clock
    if not quiet:
        print(f"GPU warmup ({gpu_warmup_secs}s) ...", end=" ", flush=True)
    warmup_end = time.time() + gpu_warmup_secs
    wa = torch.randn(4096, 4096, device="cuda", dtype=torch_dtype)
    wb = torch.randn(4096, 4096, device="cuda", dtype=torch_dtype)
    wc = torch.randn(4096, 4096, device="cuda", dtype=torch_dtype)
    while time.time() < warmup_end:
        torch.mm(wa, wb, out=wc)
    torch.cuda.synchronize()
    del wa, wb, wc
    torch.cuda.empty_cache()
    if not quiet:
        print("done")

    # Phase 1: coarse sweep
    shapes = _phase1_shapes(m_max, step=m_step, n=n, k=k)
    shapes = [(m, nn, kk) for m, nn, kk in shapes if m >= m_min]
    if not quiet:
        print(f"\nPhase 1: {len(shapes)} shapes ...")

    results = []
    for i, (M, N, K) in enumerate(shapes):
        mean_tf, med_tf, max_tf = _benchmark_shape(
            M, N, K, torch_dtype, warmup, n_iter, cache_buf, use_tf32)
        results.append((M, N, K, mean_tf, med_tf, max_tf))
        if not quiet and ((i + 1) % 5 == 0 or i == len(shapes) - 1):
            print(f"  [{i+1}/{len(shapes)}] M={M:>5}  median={med_tf:.1f} TFLOPS")

    # Phase 2: fine sweep around top-K
    results.sort(key=lambda x: x[4], reverse=True)
    top_ms = [r[0] for r in results[:top_k]]
    fine_shapes = _phase2_shapes(top_ms, n=n, k=k, radius=phase2_radius, step=phase2_step)
    existing = {(r[0], r[1], r[2]) for r in results}
    fine_shapes = [(m, nn, kk) for m, nn, kk in fine_shapes
                   if m_min <= m <= m_max and (m, nn, kk) not in existing]

    if fine_shapes:
        if not quiet:
            print(f"\nPhase 2: {len(fine_shapes)} shapes around top M values {top_ms} ...")
        for i, (M, N, K) in enumerate(fine_shapes):
            mean_tf, med_tf, max_tf = _benchmark_shape(
                M, N, K, torch_dtype, warmup, n_iter, cache_buf, use_tf32)
            results.append((M, N, K, mean_tf, med_tf, max_tf))
            if not quiet and ((i + 1) % 5 == 0 or i == len(fine_shapes) - 1):
                print(f"  [{i+1}/{len(fine_shapes)}] M={M:>5}  median={med_tf:.1f} TFLOPS")

    # Final ranking
    results.sort(key=lambda x: x[4], reverse=True)
    best = results[0]
    best_tflops = best[4]  # median

    calibration = None
    if theoretical_tflops and theoretical_tflops > 0:
        calibration = best_tflops / theoretical_tflops

    if not quiet:
        top = results[:top_k]
        print(f"\n{'='*70}")
        print(f"Top {len(top)} shapes by median TFLOPS")
        print(f"{'='*70}")

        headers = ["M", "N", "K", "mean TF", "median TF", "max TF"]
        if theoretical_tflops:
            headers.append("% peak")
        rows = []
        for M, N, K, mean_tf, med_tf, max_tf in top:
            row = [M, N, K, f"{mean_tf:.1f}", f"{med_tf:.1f}", f"{max_tf:.1f}"]
            if theoretical_tflops:
                row.append(f"{med_tf / theoretical_tflops * 100:.1f}%")
            rows.append(row)
        print(tabulate(rows, headers=headers, tablefmt="github"))

        print(f"\nBest: M={best[0]}, N={best[1]}, K={best[2]}  →  {best[4]:.1f} TFLOPS (median)")
        if theoretical_tflops:
            print(f"Theoretical peak ({dtype}): {theoretical_tflops:.0f} TFLOPS  →  "
                  f"{best[4]/theoretical_tflops*100:.1f}% utilization")
        print(f"{'='*70}")

    return {
        "gpu_name": gpu_name,
        "dtype": dtype,
        "best_tflops": best_tflops,
        "best_shape": (best[0], best[1], best[2]),
        "theoretical_tflops": theoretical_tflops,
        "calibration": calibration,
        "all_results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Matmul peak FLOPS finder")
    parser.add_argument("--dtype", default="bf16",
                        choices=sorted(set(_DTYPE_MAP.keys())),
                        help="Matmul data type (default: bf16)")
    parser.add_argument("--n", type=int, default=4096, help="N dimension (default: 4096)")
    parser.add_argument("--k", type=int, default=4096, help="K dimension (default: 4096)")
    parser.add_argument("--m-min", type=int, default=256, help="Minimum M dimension (default: 256)")
    parser.add_argument("--m-max", type=int, default=None, help="Maximum M dimension (auto from GPU memory)")
    parser.add_argument("--m-step", type=int, default=512, help="Phase 1 M step (default: 512)")
    parser.add_argument("--phase2-radius", type=int, default=512,
                        help="Phase 2 search radius around top M values (default: 512)")
    parser.add_argument("--phase2-step", type=int, default=128, help="Phase 2 M step (default: 128)")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup iterations per shape (default: 30)")
    parser.add_argument("--iter", type=int, default=100, help="Measurement iterations per shape (default: 100)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top shapes to report and refine (default: 5)")
    parser.add_argument("--cache-mb", type=int, default=256, help="L2 flush buffer size in MB (default: 256)")
    parser.add_argument("--gpu-warmup-secs", type=int, default=5,
                        help="GPU steady-state warmup in seconds (default: 5)")
    args = parser.parse_args()

    run(
        dtype=args.dtype, n=args.n, k=args.k,
        m_min=args.m_min, m_max=args.m_max, m_step=args.m_step,
        phase2_radius=args.phase2_radius, phase2_step=args.phase2_step,
        warmup=args.warmup, n_iter=args.iter, top_k=args.top_k,
        cache_mb=args.cache_mb, gpu_warmup_secs=args.gpu_warmup_secs,
    )


if __name__ == "__main__":
    main()
