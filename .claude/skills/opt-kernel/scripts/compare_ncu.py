#!/usr/bin/env python3
"""Compare two NCU reports side-by-side to quantify optimization impact.

Usage:
    python compare_ncu.py \
      --report1 <baseline.ncu-rep> --tag1 baseline \
      --report2 <optimized.ncu-rep> --tag2 optimized \
      [--output <dir>]

Outputs:
    - compare_<tag1>_vs_<tag2>.txt  — side-by-side metric comparison
    - compare_<tag1>_vs_<tag2>.json — machine-readable delta
"""

import argparse
import json
import os

from analyze_ncu import (
    load_action,
    _metric_vals,
    _metric_val,
    ALL_METRIC_GROUPS,
    classify_bottleneck,
    compute_bandwidth_analysis,
    detect_gpu,
)


def compare_reports(report1_path, tag1, report2_path, tag2, kernel_name=None):
    """Load two reports and compute metric deltas."""
    action1 = load_action(report1_path, kernel_name)
    action2 = load_action(report2_path, kernel_name)

    # Extract all metrics from both actions
    metrics1 = {}
    metrics2 = {}
    for group in ALL_METRIC_GROUPS:
        metrics1.update(_metric_vals(action1, group))
        metrics2.update(_metric_vals(action2, group))

    # Compute deltas for all metrics present in both
    deltas = {}
    for key in set(metrics1) | set(metrics2):
        v1 = metrics1.get(key)
        v2 = metrics2.get(key)
        if v1 is not None and v2 is not None:
            if v1 != 0:
                pct_change = ((v2 - v1) / abs(v1)) * 100
            else:
                pct_change = float('inf') if v2 != 0 else 0
            deltas[key] = {
                "baseline": v1,
                "optimized": v2,
                "delta": v2 - v1,
                "pct_change": pct_change,
            }

    # Bandwidth analysis
    gpu_name = detect_gpu()
    bw1 = compute_bandwidth_analysis(metrics1, gpu_name)
    bw2 = compute_bandwidth_analysis(metrics2, gpu_name)

    # Bottleneck classification
    bn1 = classify_bottleneck(metrics1, bw1)
    bn2 = classify_bottleneck(metrics2, bw2)

    return metrics1, metrics2, deltas, bw1, bw2, bn1, bn2


def format_comparison(tag1, tag2, metrics1, metrics2, deltas, bw1, bw2, bn1, bn2):
    """Format side-by-side comparison."""
    w = max(len(tag1), len(tag2), 8) + 2
    lines = []
    lines.append(f"=== NCU Report Comparison ===")
    lines.append(f"  Baseline:  {tag1}")
    lines.append(f"  Optimized: {tag2}")
    lines.append("")

    # Timing
    dur1 = metrics1.get("Duration (ns)")
    dur2 = metrics2.get("Duration (ns)")
    if dur1 and dur2:
        speedup = dur1 / dur2
        lines.append(f"--- Timing ---")
        lines.append(f"  {'Metric':<35s} {tag1:>{w}s} {tag2:>{w}s} {'Delta':>{w}s} {'Change':>8s}")
        lines.append(f"  {'Duration (us)':<35s} "
                     f"{dur1/1000:>{w}.2f} {dur2/1000:>{w}.2f} "
                     f"{(dur2-dur1)/1000:>{w}.2f} {(dur2-dur1)/dur1*100:>7.1f}%")
        lines.append(f"  Speedup: {speedup:.3f}x")
        lines.append("")

    # Bandwidth
    if bw1 or bw2:
        lines.append(f"--- Memory Bandwidth ---")
        lines.append(f"  {'Metric':<35s} {tag1:>{w}s} {tag2:>{w}s} {'Change':>8s}")
        if bw1 and bw2:
            lines.append(f"  {'Effective BW (GB/s)':<35s} "
                         f"{bw1['effective_bw_gbs']:>{w}.1f} {bw2['effective_bw_gbs']:>{w}.1f} "
                         f"{(bw2['effective_bw_gbs']-bw1['effective_bw_gbs'])/bw1['effective_bw_gbs']*100:>+7.1f}%")
            if bw1.get("bw_utilization_pct") and bw2.get("bw_utilization_pct"):
                lines.append(f"  {'BW Utilization %':<35s} "
                             f"{bw1['bw_utilization_pct']:>{w}.1f} {bw2['bw_utilization_pct']:>{w}.1f} "
                             f"{bw2['bw_utilization_pct']-bw1['bw_utilization_pct']:>+7.1f}pp")
            if bw1.get("lts_mb") and bw2.get("lts_mb"):
                lines.append(f"  {'LTS Data (MB)':<35s} "
                             f"{bw1['lts_mb']:>{w}.2f} {bw2['lts_mb']:>{w}.2f} "
                             f"{(bw2['lts_mb']-bw1['lts_mb'])/bw1['lts_mb']*100:>+7.1f}%")
        lines.append("")

    # Throughput
    lines.append(f"--- Throughput (SOL %) ---")
    lines.append(f"  {'Metric':<35s} {tag1:>{w}s} {tag2:>{w}s} {'Change':>8s}")
    for key in ["SM Throughput %", "LTS Throughput %", "L1TEX Throughput %"]:
        d = deltas.get(key)
        if d:
            lines.append(f"  {key:<35s} "
                         f"{d['baseline']:>{w}.1f} {d['optimized']:>{w}.1f} "
                         f"{d['pct_change']:>+7.1f}%")
    lines.append("")

    # Launch config
    lines.append(f"--- Launch Configuration ---")
    lines.append(f"  {'Metric':<35s} {tag1:>{w}s} {tag2:>{w}s} {'Change':>8s}")
    for key in ["Block Size", "Registers/Thread", "Waves/SM",
                "Theoretical Blocks/SM", "Theoretical Warps/SM",
                "Shared Mem/Block (bytes)"]:
        d = deltas.get(key)
        if d:
            fmt = ".1f" if isinstance(d['baseline'], float) else ""
            lines.append(f"  {key:<35s} "
                         f"{d['baseline']:>{w}{fmt}} {d['optimized']:>{w}{fmt}} "
                         f"{d['pct_change']:>+7.1f}%")
    lines.append("")

    # Stalls
    lines.append(f"--- Stall Reasons ---")
    lines.append(f"  {'Metric':<35s} {tag1:>{w}s} {tag2:>{w}s} {'Change':>8s}")
    stall_keys = [k for k in deltas if k.startswith("Stall ")]
    stall_keys.sort(key=lambda k: deltas[k].get('baseline', 0) or 0, reverse=True)
    for key in stall_keys[:8]:
        d = deltas[key]
        if d['baseline'] is not None and d['optimized'] is not None:
            lines.append(f"  {key:<35s} "
                         f"{d['baseline']:>{w}.1f} {d['optimized']:>{w}.1f} "
                         f"{d['pct_change']:>+7.1f}%")
    lines.append("")

    # Bottleneck classification
    lines.append(f"--- Bottleneck Classification ---")
    lines.append(f"  {tag1}: {bn1['classification']} "
                 f"(top stall: {bn1['top_stall']['reason']} = {bn1['top_stall']['pct']:.1f}%)")
    lines.append(f"  {tag2}: {bn2['classification']} "
                 f"(top stall: {bn2['top_stall']['reason']} = {bn2['top_stall']['pct']:.1f}%)")
    if bn1['classification'] != bn2['classification']:
        lines.append(f"  Bottleneck shifted: {bn1['classification']} -> {bn2['classification']}")
    if bw1 and bw2 and bw1.get("bw_utilization_pct") and bw2.get("bw_utilization_pct"):
        lines.append(f"  BW utilization: {bw1['bw_utilization_pct']:.1f}% -> {bw2['bw_utilization_pct']:.1f}%")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare two NCU reports")
    parser.add_argument("--report1", required=True, help="Baseline .ncu-rep file")
    parser.add_argument("--tag1", required=True, help="Label for baseline")
    parser.add_argument("--report2", required=True, help="Optimized .ncu-rep file")
    parser.add_argument("--tag2", required=True, help="Label for optimized")
    parser.add_argument("--kernel-name", default=None,
                        help="Kernel name to compare (substring match)")
    parser.add_argument("--output", default=".", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    metrics1, metrics2, deltas, bw1, bw2, bn1, bn2 = compare_reports(
        args.report1, args.tag1, args.report2, args.tag2, args.kernel_name
    )

    # Save JSON
    json_path = os.path.join(args.output, f"compare_{args.tag1}_vs_{args.tag2}.json")
    with open(json_path, "w") as f:
        json.dump({
            "baseline_tag": args.tag1,
            "optimized_tag": args.tag2,
            "deltas": deltas,
            "bandwidth": {
                "baseline": bw1,
                "optimized": bw2,
            },
            "baseline_bottleneck": bn1,
            "optimized_bottleneck": bn2,
            "duration_ns": {
                "baseline": metrics1.get("Duration (ns)"),
                "optimized": metrics2.get("Duration (ns)"),
            }
        }, f, indent=2)
    print(f"Saved comparison JSON to {json_path}")

    # Save text report
    report_text = format_comparison(
        args.tag1, args.tag2, metrics1, metrics2, deltas, bw1, bw2, bn1, bn2)
    txt_path = os.path.join(args.output, f"compare_{args.tag1}_vs_{args.tag2}.txt")
    with open(txt_path, "w") as f:
        f.write(report_text)
    print(f"Saved comparison to {txt_path}")

    # Print summary
    print(f"\n{report_text}")


if __name__ == "__main__":
    main()
