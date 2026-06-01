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
import sys

# Import from analyze_ncu
from analyze_ncu import load_action, safe, KEY_METRICS, classify_bottleneck


def compare_reports(report1_path, tag1, report2_path, tag2):
    """Load two reports and compute metric deltas."""
    action1 = load_action(report1_path)
    action2 = load_action(report2_path)

    metrics1 = {}
    metrics2 = {}
    deltas = {}

    for friendly_name, ncu_name in KEY_METRICS.items():
        v1 = safe(action1, ncu_name)
        v2 = safe(action2, ncu_name)
        metrics1[friendly_name] = v1
        metrics2[friendly_name] = v2

        if v1 is not None and v2 is not None:
            if v1 != 0:
                pct_change = ((v2 - v1) / abs(v1)) * 100
            else:
                pct_change = float('inf') if v2 != 0 else 0
            deltas[friendly_name] = {
                "baseline": v1,
                "optimized": v2,
                "delta": v2 - v1,
                "pct_change": pct_change,
            }

    bottleneck1 = classify_bottleneck(metrics1)
    bottleneck2 = classify_bottleneck(metrics2)

    return metrics1, metrics2, deltas, bottleneck1, bottleneck2


def format_comparison(tag1, tag2, metrics1, metrics2, deltas, bn1, bn2):
    """Format side-by-side comparison."""
    w = max(len(tag1), len(tag2), 8) + 2
    lines = []
    lines.append(f"=== NCU Report Comparison ===")
    lines.append(f"  Baseline:  {tag1}")
    lines.append(f"  Optimized: {tag2}")
    lines.append("")

    # Duration
    dur1 = metrics1.get("Duration (ns)")
    dur2 = metrics2.get("Duration (ns)")
    if dur1 and dur2:
        speedup = dur1 / dur2
        lines.append(f"--- Timing ---")
        lines.append(f"  {'Metric':<35s} {tag1:>{w}s} {tag2:>{w}s} {'Delta':>{w}s} {'Change':>8s}")
        lines.append(f"  {'Duration (us)':<35s} {dur1/1000:>{w}.2f} {dur2/1000:>{w}.2f} {(dur2-dur1)/1000:>{w}.2f} {(dur2-dur1)/dur1*100:>7.1f}%")
        lines.append(f"  Speedup: {speedup:.3f}x")
        lines.append("")

    # Throughput metrics
    lines.append(f"--- Throughput ---")
    lines.append(f"  {'Metric':<35s} {tag1:>{w}s} {tag2:>{w}s} {'Change':>8s}")
    for key in ["SM Throughput %", "Compute Memory Throughput %", "LTS Throughput %",
                "DRAM Throughput %", "L1TEX Throughput %"]:
        d = deltas.get(key)
        if d:
            lines.append(f"  {key:<35s} {d['baseline']:>{w}.1f} {d['optimized']:>{w}.1f} {d['pct_change']:>+7.1f}%")
    lines.append("")

    # Occupancy
    lines.append(f"--- Occupancy ---")
    lines.append(f"  {'Metric':<35s} {tag1:>{w}s} {tag2:>{w}s} {'Change':>8s}")
    for key in ["Achieved Occupancy %", "Waves/SM", "Registers/Thread"]:
        d = deltas.get(key)
        if d:
            fmt = ".1f" if isinstance(d['baseline'], float) else "d"
            lines.append(f"  {key:<35s} {d['baseline']:>{w}{fmt}} {d['optimized']:>{w}{fmt}} {d['pct_change']:>+7.1f}%")
    lines.append("")

    # Stalls
    lines.append(f"--- Stall Reasons ---")
    lines.append(f"  {'Metric':<35s} {tag1:>{w}s} {tag2:>{w}s} {'Change':>8s}")
    stall_keys = [k for k in deltas if k.startswith("Stall ")]
    stall_keys.sort(key=lambda k: deltas[k].get('baseline', 0) or 0, reverse=True)
    for key in stall_keys[:8]:
        d = deltas[key]
        if d['baseline'] is not None and d['optimized'] is not None:
            lines.append(f"  {key:<35s} {d['baseline']:>{w}.1f} {d['optimized']:>{w}.1f} {d['pct_change']:>+7.1f}%")
    lines.append("")

    # Bottleneck classification
    lines.append(f"--- Bottleneck Classification ---")
    lines.append(f"  {tag1}: {bn1['classification']} (top stall: {bn1['top_stall']['reason']} = {bn1['top_stall']['pct']:.1f}%)")
    lines.append(f"  {tag2}: {bn2['classification']} (top stall: {bn2['top_stall']['reason']} = {bn2['top_stall']['pct']:.1f}%)")
    if bn1['classification'] != bn2['classification']:
        lines.append(f"  Bottleneck shifted: {bn1['classification']} -> {bn2['classification']}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare two NCU reports")
    parser.add_argument("--report1", required=True, help="Baseline .ncu-rep file")
    parser.add_argument("--tag1", required=True, help="Label for baseline")
    parser.add_argument("--report2", required=True, help="Optimized .ncu-rep file")
    parser.add_argument("--tag2", required=True, help="Label for optimized")
    parser.add_argument("--output", default=".", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    metrics1, metrics2, deltas, bn1, bn2 = compare_reports(
        args.report1, args.tag1, args.report2, args.tag2
    )

    # Save JSON
    json_path = os.path.join(args.output, f"compare_{args.tag1}_vs_{args.tag2}.json")
    with open(json_path, "w") as f:
        json.dump({
            "baseline_tag": args.tag1,
            "optimized_tag": args.tag2,
            "deltas": deltas,
            "baseline_bottleneck": bn1,
            "optimized_bottleneck": bn2,
            "duration_ns": {
                "baseline": metrics1.get("Duration (ns)"),
                "optimized": metrics2.get("Duration (ns)"),
            }
        }, f, indent=2)
    print(f"Saved comparison JSON to {json_path}")

    # Save text report
    report_text = format_comparison(args.tag1, args.tag2, metrics1, metrics2, deltas, bn1, bn2)
    txt_path = os.path.join(args.output, f"compare_{args.tag1}_vs_{args.tag2}.txt")
    with open(txt_path, "w") as f:
        f.write(report_text)
    print(f"Saved comparison to {txt_path}")

    # Print summary
    print(f"\n{report_text}")


if __name__ == "__main__":
    main()
