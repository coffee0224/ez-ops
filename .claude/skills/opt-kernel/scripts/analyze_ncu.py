#!/usr/bin/env python3
"""Analyze a single NCU report and extract key metrics for kernel optimization.

Usage:
    python analyze_ncu.py --report <path.ncu-rep> --tag <label> [--output <dir>]

Outputs:
    - metrics_key_<tag>.json  — curated key metrics (machine-readable)
    - metrics_key_<tag>.txt   — human-readable summary
    - analysis_<tag>.txt      — diagnostic summary with bottleneck classification
"""

import argparse
import json
import os
import sys
from pathlib import Path

# --- NCU report loading ---

def locate_ncu_report():
    """Find the ncu_report Python module from common CUDA install paths."""
    try:
        import ncu_report
        return ncu_report
    except ImportError:
        pass

    import glob
    search_paths = [
        "/usr/local/cuda-*/nsight-compute-*/extras/python",
        "/usr/local/cuda/nsight-compute-*/extras/python",
        "/opt/nvidia/nsight-compute-*/extras/python",
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
    print("ERROR: Cannot find ncu_report module. Set PYTHONPATH to your Nsight Compute extras/python directory.")
    sys.exit(1)


def load_action(report_path):
    """Load an NCU report and return the first kernel action."""
    ncu_report = locate_ncu_report()
    report = ncu_report.load_report(report_path)
    action = report[0][0]  # first range, first action
    return action


# --- Safe metric access ---

def safe(action, name, default=None):
    """Read a metric value, returning default if missing."""
    try:
        return action[name].value()
    except Exception:
        return default


def safe_many(action, names, default=None):
    """Read multiple metrics into a dict."""
    return {name: safe(action, name, default) for name in names}


# --- Key metric definitions ---

KEY_METRICS = {
    # Timing
    "Duration (ns)": "gpu__time_duration.sum",
    # Throughput SOL
    "SM Throughput %": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "Compute Memory Throughput %": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "L1TEX Throughput %": "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "LTS Throughput %": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    # Occupancy
    "Grid Size": "launch__grid_size",
    "Block Size": "launch__block_size",
    "Waves/SM": "launch__waves_per_multiprocessor",
    "Registers/Thread": "launch__registers_per_thread",
    "Shared Mem/Block (bytes)": "launch__shared_memory_per_block",
    "Theoretical Occupancy %": "sm__warps_active.avg.pct_of_peak",
    "Achieved Occupancy %": "smsp__warps_active.avg.pct_of_peak",
    # DRAM
    "DRAM Read Bytes": "memory_l1_hierarchy_read_transactions.sum",
    "DRAM Write Bytes": "memory_l1_hierarchy_write_transactions.sum",
    "DRAM Throughput %": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    # Stall reasons (top-level)
    "Stall Long Scoreboard %": "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_long_scoreboard",
    "Stall Wait %": "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_wait",
    "Stall Short Scoreboard %": "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_short_scoreboard",
    "Stall Math Pipe %": "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_math_pipe_throttle",
    "Stall MIO %": "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_mio_throttle",
    "Stall LG %": "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_lg_throttle",
    "Stall Not Selected %": "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_not_selected",
    "Stall Barrier %": "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_barrier",
    "Stall Branch %": "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_branch_resolving",
    "Stall No Instruction %": "smsp__issue_active.avg.pct_of_peak_sustained_active_stalled_no_instruction",
    # IPC
    "Executed IPC": "smsp__issue_active.avg.inst_per_cycle",
    # Memory instructions
    "Global Load Insts": "smsp__inst_executed.sum.op_global_load",
    "Global Store Insts": "smsp__inst_executed.sum.op_global_store",
    "Shared Load Insts": "smsp__inst_executed.sum.op_shared_load",
    "Shared Store Insts": "smsp__inst_executed.sum.op_shared_store",
}


def classify_bottleneck(metrics):
    """Classify the kernel bottleneck based on key metrics."""
    sm_throughput = metrics.get("SM Throughput %", 0) or 0
    lts_throughput = metrics.get("LTS Throughput %", 0) or 0
    dram_throughput = metrics.get("DRAM Throughput %", 0) or 0
    occupancy = metrics.get("Achieved Occupancy %", 0) or 0
    ipc = metrics.get("Executed IPC", 0) or 0

    # Find top stall reason
    stall_keys = [k for k in metrics if k.startswith("Stall ")]
    stall_pairs = [(k, metrics[k] or 0) for k in stall_keys]
    stall_pairs.sort(key=lambda x: x[1], reverse=True)
    top_stall = stall_pairs[0] if stall_pairs else ("None", 0)

    memory_pressure = max(lts_throughput, dram_throughput)
    compute_pressure = sm_throughput

    if memory_pressure > 60 and compute_pressure < 50:
        classification = "MEMORY_BOUND"
    elif compute_pressure > 60 and memory_pressure < 50:
        classification = "COMPUTE_BOUND"
    elif occupancy < 30 or (top_stall[1] > 30 and top_stall[0] in ("Stall Not Selected %", "Stall No Instruction %")):
        classification = "OCCUPANCY_BOUND"
    else:
        classification = "BALANCED"

    return {
        "classification": classification,
        "sm_throughput": sm_throughput,
        "memory_throughput": memory_pressure,
        "occupancy": occupancy,
        "ipc": ipc,
        "top_stall": {"reason": top_stall[0], "pct": top_stall[1]},
        "second_stall": {"reason": stall_pairs[1][0], "pct": stall_pairs[1][1]} if len(stall_pairs) > 1 else None,
    }


def generate_recommendations(bottleneck):
    """Generate optimization recommendations based on bottleneck classification."""
    recs = []
    cls = bottleneck["classification"]

    if cls == "MEMORY_BOUND":
        recs.append("Vectorize memory accesses (uint4/float4 loads)")
        recs.append("Improve memory coalescing (sequential access per warp)")
        recs.append("Use shared memory for data reuse")
        recs.append("Consider data prefetching or async memory copies")
        stall = bottleneck["top_stall"]["reason"]
        if "long_scoreboard" in stall:
            recs.append("Reduce dependency chain length between loads and uses")
        if "lg_throttle" in stall:
            recs.append("Reduce L1/LSU pressure, consider caching hints")

    elif cls == "COMPUTE_BOUND":
        recs.append("Use tensor cores (HMMA) if applicable")
        recs.append("Fuse multiple operations to reduce data movement")
        recs.append("Check for unnecessary type conversions")
        recs.append("Reduce instruction count through algorithmic optimization")

    elif cls == "OCCUPANCY_BOUND":
        recs.append("Reduce register usage (check compilation with -Xptxas -v)")
        recs.append("Reduce shared memory per block")
        recs.append("Increase grid size or blocks per launch")
        recs.append("Consider persistent kernel pattern for small workloads")

    else:  # BALANCED
        recs.append("Profile with --set source for per-line stall analysis")
        recs.append("Look at specific hotspots with extract_stall_hotspots.py")
        recs.append("Consider warp-specialization for pipeline optimization")

    return recs


def format_report(tag, metrics, bottleneck, recommendations):
    """Format the analysis as a human-readable report."""
    lines = []
    lines.append(f"=== NCU Analysis: {tag} ===\n")

    lines.append("--- Kernel Timing ---")
    dur_ns = metrics.get("Duration (ns)", 0)
    if dur_ns:
        lines.append(f"  Duration: {dur_ns:.0f} ns ({dur_ns/1000:.2f} us)")

    lines.append(f"\n--- Throughput (SOL %) ---")
    for k in ["SM Throughput %", "Compute Memory Throughput %", "LTS Throughput %",
              "DRAM Throughput %", "L1TEX Throughput %"]:
        v = metrics.get(k)
        if v is not None:
            lines.append(f"  {k}: {v:.1f}%")

    lines.append(f"\n--- Launch Configuration ---")
    for k in ["Grid Size", "Block Size", "Waves/SM", "Registers/Thread",
              "Shared Mem/Block (bytes)", "Theoretical Occupancy %", "Achieved Occupancy %"]:
        v = metrics.get(k)
        if v is not None:
            label = k.replace(" (bytes)", "")
            if isinstance(v, float):
                lines.append(f"  {label}: {v:.1f}")
            else:
                lines.append(f"  {label}: {v}")

    lines.append(f"\n--- Top Stall Reasons ---")
    stall_keys = [k for k in metrics if k.startswith("Stall ")]
    stall_pairs = [(k, metrics[k] or 0) for k in stall_keys]
    stall_pairs.sort(key=lambda x: x[1], reverse=True)
    for name, val in stall_pairs[:5]:
        if val and val > 0.1:
            lines.append(f"  {name}: {val:.1f}%")

    lines.append(f"\n--- Bottleneck Classification ---")
    lines.append(f"  Classification: {bottleneck['classification']}")
    lines.append(f"  Top stall: {bottleneck['top_stall']['reason']} = {bottleneck['top_stall']['pct']:.1f}%")

    lines.append(f"\n--- Optimization Recommendations ---")
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"  {i}. {rec}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze NCU report for kernel optimization")
    parser.add_argument("--report", required=True, help="Path to .ncu-rep file")
    parser.add_argument("--tag", required=True, help="Short label for this report")
    parser.add_argument("--output", default=".", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading report: {args.report}")
    action = load_action(args.report)

    # Extract metrics
    metrics = {}
    for friendly_name, ncu_name in KEY_METRICS.items():
        val = safe(action, ncu_name)
        metrics[friendly_name] = val

    # Classify bottleneck
    bottleneck = classify_bottleneck(metrics)
    recommendations = generate_recommendations(bottleneck)

    # Save JSON
    json_path = os.path.join(args.output, f"metrics_key_{args.tag}.json")
    with open(json_path, "w") as f:
        json.dump({
            "tag": args.tag,
            "metrics": {k: v for k, v in metrics.items() if v is not None},
            "bottleneck": bottleneck,
            "recommendations": recommendations,
        }, f, indent=2)
    print(f"Saved metrics to {json_path}")

    # Save human-readable report
    report_text = format_report(args.tag, metrics, bottleneck, recommendations)
    txt_path = os.path.join(args.output, f"metrics_key_{args.tag}.txt")
    with open(txt_path, "w") as f:
        f.write(report_text)
    print(f"Saved report to {txt_path}")

    # Save analysis
    analysis_path = os.path.join(args.output, f"analysis_{args.tag}.txt")
    with open(analysis_path, "w") as f:
        f.write(report_text)
    print(f"Saved analysis to {analysis_path}")

    # Print summary
    print(f"\n{report_text}")


if __name__ == "__main__":
    main()
