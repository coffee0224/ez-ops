#!/usr/bin/env python3
"""Analyze a single NCU report and extract key metrics for kernel optimization.

Usage:
    python analyze_ncu.py --report <path.ncu-rep> --tag <label> [--output <dir>]
    python analyze_ncu.py --report <path.ncu-rep> --tag <label> --kernel-name gemv_ws_kernel

Outputs:
    - metrics_key_<tag>.json  — curated key metrics (machine-readable)
    - metrics_key_<tag>.txt   — human-readable summary
    - analysis_<tag>.txt      — diagnostic summary with bottleneck classification
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# NCU report loading
# ---------------------------------------------------------------------------

def locate_ncu_report():
    """Find the ncu_report Python module from common CUDA install paths."""
    try:
        import ncu_report
        return ncu_report
    except ImportError:
        pass

    import glob
    search_paths = [
        "/opt/nvidia/nsight-compute-*/extras/python",
        "/usr/local/cuda-*/nsight-compute-*/extras/python",
        "/usr/local/cuda/nsight-compute-*/extras/python",
        "/opt/cuda/nsight-compute-*/extras/python",
    ]
    for pattern in search_paths:
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches:
            sys.path.insert(0, matches[0])
            try:
                import ncu_report
                return ncu_report
            except ImportError:
                continue
    print("ERROR: Cannot find ncu_report module. "
          "Set PYTHONPATH to your Nsight Compute extras/python directory.")
    sys.exit(1)


def list_kernels(report_path):
    """List all kernels in an NCU report. Returns [(name, action_idx)]."""
    ncu_report = locate_ncu_report()
    report = ncu_report.load_report(report_path)
    kernels = []
    for ri in range(report.num_ranges()):
        rng = report.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            act = rng.action_by_idx(ai)
            kernels.append((act.name(), ri, ai))
    return kernels


def load_action(report_path, kernel_name=None):
    """Load an NCU report and return a kernel action.

    If kernel_name is given, find the matching kernel (substring match).
    Otherwise, pick the kernel with the longest duration (likely the target).
    """
    ncu_report = locate_ncu_report()
    report = ncu_report.load_report(report_path)

    # Collect all kernel actions
    candidates = []
    for ri in range(report.num_ranges()):
        rng = report.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            act = rng.action_by_idx(ai)
            candidates.append((act.name(), ri, ai, act))

    if not candidates:
        print("ERROR: No kernels found in report.")
        sys.exit(1)

    # Filter by name if specified
    if kernel_name:
        matches = [(n, ri, ai, a) for n, ri, ai, a in candidates
                    if kernel_name in n]
        if not matches:
            print(f"ERROR: Kernel '{kernel_name}' not found. Available:")
            for n, ri, ai, _ in candidates:
                print(f"  [{ai}] {n}")
            sys.exit(1)
        if len(matches) > 1:
            print(f"WARNING: {len(matches)} kernels match '{kernel_name}', using first.")
        _, ri, ai, act = matches[0]
        return act

    # Auto-select: pick the longest-duration kernel
    best = None
    best_dur = -1
    for name, ri, ai, act in candidates:
        dur = _metric_val(act, "gpu__time_duration.sum")
        if dur is not None and dur > best_dur:
            best_dur = dur
            best = (name, ri, ai, act)

    if best is None:
        # Fallback: try first action
        name, ri, ai, act = candidates[0]
        print(f"WARNING: Could not determine kernel durations, using first: {name}")
        return act

    name, ri, ai, act = best
    print(f"Auto-selected kernel: {name} (duration={best_dur:.0f} ns)")
    return act


# ---------------------------------------------------------------------------
# Safe metric access
# ---------------------------------------------------------------------------

def _metric_val(action, name):
    """Read a single metric value, return None if missing."""
    try:
        m = action.metric_by_name(name)
        if m is None:
            return None
        return m.value()
    except Exception:
        return None


def _metric_vals(action, names):
    """Read multiple metrics into a dict, skipping None values."""
    result = {}
    for friendly, ncu_name in names:
        v = _metric_val(action, ncu_name)
        if v is not None:
            result[friendly] = v
    return result


# ---------------------------------------------------------------------------
# GPU profiles for BW calculation
# ---------------------------------------------------------------------------

GPU_PROFILES = {
    "NVIDIA GeForce RTX 5060 Ti": {"hbm_peak_gbs": 448, "hbm_calib": 0.845},
    "NVIDIA GeForce RTX 4090": {"hbm_peak_gbs": 1008, "hbm_calib": 0.88},
    "NVIDIA H100 80GB HBM3": {"hbm_peak_gbs": 3350, "hbm_calib": 0.86},
    "NVIDIA H200": {"hbm_peak_gbs": 4800, "hbm_calib": 0.86},
    "NVIDIA B200": {"hbm_peak_gbs": 8000, "hbm_calib": 0.85},
}


def detect_gpu():
    """Detect GPU name via nvidia-smi."""
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip().split("\n")[0].strip()
    except FileNotFoundError:
        pass
    return None


# ---------------------------------------------------------------------------
# Metric definitions — organized by category
# ---------------------------------------------------------------------------

# Each tuple: (friendly_name, ncu_metric_name)
TIMING_METRICS = [
    ("Duration (ns)", "gpu__time_duration.sum"),
]

THROUGHPUT_METRICS = [
    ("SM Throughput %", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("LTS Throughput %", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("L1TEX Throughput %", "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"),
]

LAUNCH_METRICS = [
    ("Grid Size", "launch__grid_size"),
    ("Block Size", "launch__block_size"),
    ("Registers/Thread", "launch__registers_per_thread"),
    ("Shared Mem/Block (bytes)", "launch__shared_memory_per_block"),
    ("Shared Mem Dynamic (bytes)", "launch__shared_memory_per_block_dynamic"),
    ("Waves/SM", "launch__waves_per_multiprocessor"),
    ("Theoretical Blocks/SM", "launch__theoretical_active_blocks_per_sm"),
    ("Theoretical Warps/SM", "launch__theoretical_active_warps_per_sm"),
    ("Occupancy Limit Regs", "launch__occupancy_limit_registers"),
    ("Occupancy Limit SharedMem", "launch__occupancy_limit_shared_mem"),
    ("Occupancy Limit Warps", "launch__occupancy_limit_warps"),
]

MEMORY_METRICS = [
    ("LTS Sectors", "lts__t_sectors.sum"),
    ("LTS Requests", "lts__t_requests.sum"),
    ("L1 Global Load Sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"),
    ("L1 Global Store Sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"),
    ("L1 Shared Load Sectors", "l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum"),
    ("L1 Shared Store Sectors", "l1tex__t_sectors_pipe_lsu_mem_shared_op_st.sum"),
]

INSTRUCTION_METRICS = [
    ("Shared Load Insts", "smsp__sass_inst_executed_op_shared_ld.sum"),
    ("Shared Store Insts", "smsp__sass_inst_executed_op_shared_st.sum"),
    ("Global Load Insts", "smsp__sass_inst_executed_op_global_ld.sum"),
    ("Global Store Insts", "smsp__sass_inst_executed_op_global_st.sum"),
    ("Local Load Insts", "smsp__sass_inst_executed_op_local_ld.sum"),
    ("Local Store Insts", "smsp__sass_inst_executed_op_local_st.sum"),
]

STALL_METRICS = [
    ("Stall Long Scoreboard %", "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_long_scoreboard"),
    ("Stall Wait %", "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_wait"),
    ("Stall Short Scoreboard %", "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_short_scoreboard"),
    ("Stall Math Pipe %", "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_math_pipe_throttle"),
    ("Stall MIO %", "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_mio_throttle"),
    ("Stall LG %", "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_lg_throttle"),
    ("Stall Not Selected %", "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_not_selected"),
    ("Stall Barrier %", "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_barrier"),
    ("Stall Branch %", "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_branch_resolving"),
    ("Stall No Instruction %", "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_no_instruction"),
]

CYCLE_METRICS = [
    ("SM Cycles Elapsed", "sm__cycles_elapsed.sum"),
    ("SM Cycles Active", "sm__cycles_active.sum"),
]

ALL_METRIC_GROUPS = [
    TIMING_METRICS, THROUGHPUT_METRICS, LAUNCH_METRICS,
    MEMORY_METRICS, INSTRUCTION_METRICS, STALL_METRICS, CYCLE_METRICS,
]


# ---------------------------------------------------------------------------
# Analysis logic
# ---------------------------------------------------------------------------

def compute_bandwidth_analysis(metrics, gpu_name):
    """Compute effective memory bandwidth and utilization."""
    dur_ns = metrics.get("Duration (ns)")
    lts_sectors = metrics.get("LTS Sectors")

    if dur_ns is None or lts_sectors is None or dur_ns == 0:
        return None

    lts_bytes = lts_sectors * 32
    eff_bw_gbs = (lts_bytes / (dur_ns * 1e-9)) / 1e9

    # Find GPU peak
    peak_gbs = None
    if gpu_name:
        for key, profile in GPU_PROFILES.items():
            if key in gpu_name:
                peak_gbs = profile["hbm_peak_gbs"] * profile["hbm_calib"]
                break

    result = {
        "lts_bytes": lts_bytes,
        "lts_mb": lts_bytes / 1e6,
        "effective_bw_gbs": eff_bw_gbs,
        "peak_effective_gbs": peak_gbs,
        "bw_utilization_pct": (eff_bw_gbs / peak_gbs * 100) if peak_gbs else None,
    }
    return result


def classify_bottleneck(metrics, bw_analysis):
    """Classify the kernel bottleneck."""
    sm_throughput = metrics.get("SM Throughput %") or 0
    lts_throughput = metrics.get("LTS Throughput %") or 0
    bw_util = (bw_analysis or {}).get("bw_utilization_pct") or 0

    # Find top stall reason
    stall_pairs = [(k, v) for k, v in metrics.items()
                   if k.startswith("Stall ") and v is not None]
    stall_pairs.sort(key=lambda x: x[1], reverse=True)
    top_stall = stall_pairs[0] if stall_pairs else ("None", 0)
    second_stall = stall_pairs[1] if len(stall_pairs) > 1 else None

    # Classification: prioritize BW utilization over raw throughput %
    if bw_util > 80:
        classification = "MEMORY_BANDWIDTH_SATURATED"
    elif lts_throughput > 60 and sm_throughput < 50:
        classification = "MEMORY_BOUND"
    elif sm_throughput > 60:
        classification = "COMPUTE_BOUND"
    elif bw_util < 50 and sm_throughput < 30:
        classification = "OCCUPANCY_BOUND"
    else:
        classification = "BALANCED"

    return {
        "classification": classification,
        "sm_throughput_pct": sm_throughput,
        "lts_throughput_pct": lts_throughput,
        "bw_utilization_pct": bw_util,
        "top_stall": {"reason": top_stall[0], "pct": top_stall[1]},
        "second_stall": {"reason": second_stall[0], "pct": second_stall[1]}
                       if second_stall else None,
    }


def generate_recommendations(bottleneck):
    """Generate optimization recommendations."""
    cls = bottleneck["classification"]
    recs = []

    if cls == "MEMORY_BANDWIDTH_SATURATED":
        recs.append("Kernel is near peak HBM bandwidth — minimal headroom left")
        recs.append("Focus on reducing total bytes transferred (algorithmic changes)")
        recs.append("Check data overhead (LTS bytes vs ideal bytes)")

    elif cls == "MEMORY_BOUND":
        recs.append("Vectorize memory accesses (uint4/float4 loads)")
        recs.append("Improve memory coalescing (sequential access per warp)")
        recs.append("Use shared memory for data reuse")
        stall = bottleneck["top_stall"]["reason"]
        if "long_scoreboard" in stall:
            recs.append("Reduce dependency chain length between loads and uses")

    elif cls == "COMPUTE_BOUND":
        recs.append("Use tensor cores (HMMA/WMMA) if applicable")
        recs.append("Fuse operations to reduce data movement")
        recs.append("Check for unnecessary type conversions")

    elif cls == "OCCUPANCY_BOUND":
        recs.append("Reduce register usage (check -Xptxas -v)")
        recs.append("Reduce shared memory per block")
        recs.append("Increase grid size or blocks per launch")

    else:  # BALANCED
        recs.append("Profile with --set source for per-line stall analysis")
        recs.append("Consider warp-specialization for pipeline optimization")

    return recs


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(kernel_name, gpu_name, tag, metrics, bw_analysis, bottleneck, recs):
    """Format the analysis as a human-readable report."""
    lines = []
    lines.append(f"=== NCU Analysis: {tag} ===\n")
    lines.append(f"Kernel: {kernel_name}")
    if gpu_name:
        lines.append(f"GPU: {gpu_name}")

    # Timing
    dur_ns = metrics.get("Duration (ns)")
    if dur_ns:
        lines.append(f"\n--- Kernel Timing ---")
        lines.append(f"  Duration: {dur_ns:.0f} ns ({dur_ns/1000:.1f} us)")

    # Bandwidth analysis (most important section)
    if bw_analysis:
        lines.append(f"\n--- Memory Bandwidth ---")
        lines.append(f"  LTS data: {bw_analysis['lts_mb']:.2f} MB")
        lines.append(f"  Effective BW: {bw_analysis['effective_bw_gbs']:.1f} GB/s")
        if bw_analysis['peak_effective_gbs']:
            lines.append(f"  Peak effective BW: {bw_analysis['peak_effective_gbs']:.1f} GB/s")
            lines.append(f"  *** BW Utilization: {bw_analysis['bw_utilization_pct']:.1f}% ***")

    # Throughput
    lines.append(f"\n--- Throughput (SOL %) ---")
    for k in ["SM Throughput %", "LTS Throughput %", "L1TEX Throughput %"]:
        v = metrics.get(k)
        if v is not None:
            lines.append(f"  {k}: {v:.1f}%")

    # Launch config
    lines.append(f"\n--- Launch Configuration ---")
    for k in ["Grid Size", "Block Size", "Registers/Thread", "Waves/SM",
              "Shared Mem/Block (bytes)", "Theoretical Blocks/SM",
              "Theoretical Warps/SM"]:
        v = metrics.get(k)
        if v is not None:
            label = k.replace(" (bytes)", "")
            if isinstance(v, float):
                lines.append(f"  {label}: {v:.1f}")
            else:
                lines.append(f"  {label}: {v}")

    # Occupancy limits
    lines.append(f"\n--- Occupancy Limits ---")
    for k in ["Occupancy Limit Regs", "Occupancy Limit SharedMem",
              "Occupancy Limit Warps"]:
        v = metrics.get(k)
        if v is not None:
            lines.append(f"  {k}: {v}")

    # Instructions
    lines.append(f"\n--- Instructions ---")
    for k in ["Shared Load Insts", "Shared Store Insts",
              "Global Load Insts", "Global Store Insts"]:
        v = metrics.get(k)
        if v is not None:
            lines.append(f"  {k}: {v:.0f}")

    # Stalls
    lines.append(f"\n--- Stall Reasons ---")
    stall_pairs = [(k, v) for k, v in metrics.items()
                   if k.startswith("Stall ") and v is not None]
    stall_pairs.sort(key=lambda x: x[1], reverse=True)
    any_stalls = False
    for name, val in stall_pairs[:5]:
        if val > 0.01:
            lines.append(f"  {name}: {val:.1f}%")
            any_stalls = True
    if not any_stalls:
        lines.append("  (no significant stalls detected)")

    # Bottleneck
    lines.append(f"\n--- Bottleneck Classification ---")
    lines.append(f"  Classification: {bottleneck['classification']}")
    ts = bottleneck["top_stall"]
    lines.append(f"  Top stall: {ts['reason']} = {ts['pct']:.1f}%")
    if bw_analysis and bw_analysis.get("bw_utilization_pct"):
        lines.append(f"  BW utilization: {bw_analysis['bw_utilization_pct']:.1f}%")

    # Recommendations
    lines.append(f"\n--- Optimization Recommendations ---")
    for i, rec in enumerate(recs, 1):
        lines.append(f"  {i}. {rec}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optimization log
# ---------------------------------------------------------------------------

def append_to_log(output_dir, kernel, tag, kernel_name, gpu_name,
                  metrics, bw_analysis, bottleneck, recs, report_path):
    """Append an iteration summary to .profiles/<kernel>_opt_log.md.

    Creates the log with a header on first call, appends on subsequent calls.
    """
    log_path = os.path.join(output_dir, f"{kernel}_opt_log.md")

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write(f"# Optimization Log: {kernel}\n\n")
            f.write(f"- Kernel: {kernel_name}\n")
            f.write(f"- GPU: {gpu_name or 'unknown'}\n")
            f.write(f"- Created: {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n")

    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    lines = [f"## {tag} ({ts})\n"]

    dur_ns = metrics.get("Duration (ns)")
    if dur_ns:
        lines.append(f"- Duration: **{dur_ns/1000:.2f} us**")

    if bw_analysis:
        bw_line = f"- Effective BW: {bw_analysis['effective_bw_gbs']:.1f} GB/s"
        if bw_analysis.get('bw_utilization_pct') is not None:
            bw_line += f" ({bw_analysis['bw_utilization_pct']:.1f}% util)"
        lines.append(bw_line)

    lines.append(f"- Bottleneck: {bottleneck['classification']}")
    top = bottleneck['top_stall']
    lines.append(f"- Top stall: {top['reason']} = {top['pct']:.1f}%")

    if report_path:
        lines.append(f"- NCU report: `{os.path.basename(report_path)}`")

    if recs:
        lines.append("- Recommendations:")
        for r in recs[:3]:
            lines.append(f"  - {r}")

    lines.append("")

    with open(log_path, "a") as f:
        f.write("\n".join(lines) + "\n")

    return log_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze NCU report for kernel optimization")
    parser.add_argument("--report", required=True, help="Path to .ncu-rep file")
    parser.add_argument("--tag", required=True, help="Short label for this report")
    parser.add_argument("--kernel-name", default=None,
                        help="Kernel name to analyze (substring match). "
                             "Auto-selects longest-duration kernel if omitted.")
    parser.add_argument("--list-kernels", action="store_true",
                        help="List all kernels in the report and exit.")
    parser.add_argument("--kernel", default=None,
                        help="Op name for optimization log. When set, appends to <output>/<kernel>_opt_log.md")
    parser.add_argument("--output", default=".profiles", help="Output directory (default: .profiles)")
    args = parser.parse_args()

    # List kernels mode
    if args.list_kernels:
        print(f"Kernels in {args.report}:")
        for name, ri, ai in list_kernels(args.report):
            print(f"  [{ai}] {name}")
        return

    os.makedirs(args.output, exist_ok=True)

    # Detect GPU
    gpu_name = detect_gpu()

    # Load action
    print(f"Loading report: {args.report}")
    action = load_action(args.report, args.kernel_name)
    kernel_name = action.name()
    print(f"Analyzing kernel: {kernel_name}")

    # Extract all metrics
    metrics = {}
    for group in ALL_METRIC_GROUPS:
        metrics.update(_metric_vals(action, group))

    # Bandwidth analysis
    bw_analysis = compute_bandwidth_analysis(metrics, gpu_name)

    # Classify bottleneck
    bottleneck = classify_bottleneck(metrics, bw_analysis)
    recommendations = generate_recommendations(bottleneck)

    # Save JSON
    json_path = os.path.join(args.output, f"metrics_key_{args.tag}.json")
    with open(json_path, "w") as f:
        json.dump({
            "tag": args.tag,
            "kernel_name": kernel_name,
            "gpu": gpu_name,
            "metrics": {k: v for k, v in metrics.items() if v is not None},
            "bandwidth": bw_analysis,
            "bottleneck": bottleneck,
            "recommendations": recommendations,
        }, f, indent=2)
    print(f"Saved metrics to {json_path}")

    # Format and save report
    report_text = format_report(
        kernel_name, gpu_name, args.tag, metrics, bw_analysis, bottleneck, recommendations)
    txt_path = os.path.join(args.output, f"metrics_key_{args.tag}.txt")
    with open(txt_path, "w") as f:
        f.write(report_text)
    print(f"Saved report to {txt_path}")

    # Save analysis (same content)
    analysis_path = os.path.join(args.output, f"analysis_{args.tag}.txt")
    with open(analysis_path, "w") as f:
        f.write(report_text)

    # Print summary
    print(f"\n{report_text}")

    # Append to optimization log
    if args.kernel:
        log_path = append_to_log(
            args.output, args.kernel, args.tag, kernel_name, gpu_name,
            metrics, bw_analysis, bottleneck, recommendations, args.report)
        print(f"Appended to log: {log_path}")


if __name__ == "__main__":
    main()
