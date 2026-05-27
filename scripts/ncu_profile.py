#!/usr/bin/env python
"""Profile ez-ops kernels using NVIDIA Nsight Compute (ncu).

Usage:
  # Show available ops
  python ncu_profile.py -h

  # Show kernels and params for an op
  python ncu_profile.py vector_add -h

  # Profile reference implementation only
  python ncu_profile.py vector_add -p n=1048576

  # Profile specific kernels
  python ncu_profile.py vector_add -k triton,cuda -p n=1048576

  # Positional params
  python ncu_profile.py vector_add -k ref,triton -p 1048576 -o results/
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path


def op_name_to_class(op_name: str) -> str:
    return "".join(p.capitalize() for p in op_name.split("_")) + "Op"


def show_help_no_op() -> None:
    """Show usage and list all available ops."""
    print(textwrap.dedent("""\
        Usage: ncu_profile.py <op_name> [options]

        Profile ez-ops kernels using NVIDIA Nsight Compute (ncu).

        Options:
          -k, --kernels KERNELS  Kernel backends to profile (comma-separated, default: ref)
          -o, --output-dir DIR   Output directory (default: .profiles)
          -p, --params PARAMS    Op constructor params: 'M=5,N=10' or '5,10,12'
          -w, --warmup N         Warmup iterations (default: 10)
          -h, --help             Show this help message
    """))
    try:
        from ezops import list_ops

        ops = list_ops()
        if ops:
            print("Available ops:")
            for name in ops:
                print(f"  {name}")
            print(f"\nRun 'ncu_profile.py <op_name> -h' for details on a specific op.")
        else:
            print("No ops found.")
    except Exception as e:
        print(f"Could not list ops: {e}")


def show_help_for_op(op_name: str) -> None:
    """Show available kernels and parameter descriptions for an op."""
    class_name = op_name_to_class(op_name)

    try:
        from ezops import list_backends

        mod = importlib.import_module("ezops")
        op_cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        print(f"Error: unknown op {op_name!r}: {e}")
        sys.exit(1)

    print(f"Op: {op_name} ({class_name})")
    print()

    # Show kernels
    backends = list_backends(op_name)
    all_kernels = ["ref"] + backends
    print("Available kernels:")
    for k in all_kernels:
        print(f"  {k}")
    print()

    # Show params
    params_desc = getattr(op_cls, "_params_desc", {})
    if params_desc:
        print("Parameters:")
        for name, desc in params_desc.items():
            print(f"  {name}  {desc}")
        print()
        print(f"Example: -p {','.join(f'{k}=<' + k + '>' for k in params_desc)}")
    else:
        print("Parameters: (no description available)")

    print()
    print(f"Usage: ncu_profile.py {op_name} -k <kernels> -p <params>")


def parse_params(params_str: str) -> tuple[list, dict]:
    """Parse 'M=5,N=10' -> ([], {'M':5,'N':10}) or '5,10,12' -> ([5,10,12], {})."""
    if not params_str:
        return [], {}
    if "=" in params_str:
        kwargs = {}
        for pair in params_str.split(","):
            k, v = pair.strip().split("=")
            try:
                kwargs[k.strip()] = int(v.strip())
            except ValueError:
                kwargs[k.strip()] = float(v.strip())
        return [], kwargs
    args = []
    for v in params_str.split(","):
        v = v.strip()
        try:
            args.append(int(v))
        except ValueError:
            args.append(float(v))
    return args, {}


def _build_op_params_str(pos_args: list, kw_args: dict, backend: str | None = None) -> str:
    parts = [repr(a) for a in pos_args]
    parts += [f"{k}={v!r}" for k, v in kw_args.items()]
    if backend is not None:
        parts.append(f"backend={backend!r}")
    return ", ".join(parts)


def generate_profile_script(op_name: str, kernel_name: str, pos_args: list, kw_args: dict) -> str:
    class_name = op_name_to_class(op_name)
    backend = None if kernel_name == "ref" else kernel_name
    params_str = _build_op_params_str(pos_args, kw_args, backend)
    call_line = "op._ref_forward(*data)" if kernel_name == "ref" else "op(*data)"

    return textwrap.dedent(f"""\
        import torch
        from ezops import {class_name}

        op = {class_name}({params_str})
        data = op.gen_data()
        {call_line}
        torch.cuda.synchronize()
    """)


def warmup(op_name: str, pos_args: list, kw_args: dict, n_warmup: int):
    """Warm up the GPU by running a kernel repeatedly."""
    import importlib

    import torch

    class_name = op_name_to_class(op_name)
    mod = importlib.import_module("ezops")
    op_cls = getattr(mod, class_name)

    from ezops import list_backends

    backends = list_backends(op_name)
    backend = backends[0] if backends else "triton"

    params_str = _build_op_params_str(pos_args, kw_args, backend)
    # Use eval for simplicity since we control all inputs
    op = eval(f"op_cls({params_str})")  # noqa: S307
    data = op.gen_data()

    for _ in range(n_warmup):
        op(*data)
    torch.cuda.synchronize()

    del op, data
    torch.cuda.empty_cache()


def main():
    # Handle -h before argparse processes it, so we can show custom help
    if "-h" in sys.argv or "--help" in sys.argv:
        # Remove -h/--help from argv temporarily
        filtered = [a for a in sys.argv[1:] if a not in ("-h", "--help")]
        if not filtered:
            show_help_no_op()
        else:
            # First non-flag argument is the op name
            op_name = filtered[0]
            show_help_for_op(op_name)
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Profile ez-ops kernels using NVIDIA Nsight Compute (ncu)",
    )
    parser.add_argument("op_name", help="Op name in snake_case (e.g. vector_add)")
    parser.add_argument(
        "--kernels",
        "-k",
        default="ref",
        help="Kernel backends to profile, comma-separated (default: ref). "
        "Use 'all' to profile all registered backends.",
    )
    parser.add_argument("--output-dir", "-o", default=".profiles", help="Output directory")
    parser.add_argument(
        "--params",
        "-p",
        default="",
        help="Op constructor params: 'M=5,N=10,K=12' or '5,10,12'",
    )
    parser.add_argument("--warmup", "-w", type=int, default=10, help="Warmup iterations (default: 10)")
    args = parser.parse_args()

    # Check ncu
    ncu_path = shutil.which("ncu")
    if ncu_path is None:
        print("Error: ncu not found. Install NVIDIA Nsight Compute and add to PATH.")
        sys.exit(1)

    # Resolve kernel list
    if args.kernels == "all":
        from ezops import list_backends

        kernels = ["ref"] + list_backends(args.op_name)
    else:
        kernels = [k.strip() for k in args.kernels.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pos_args, kw_args = parse_params(args.params)

    # Verify op import
    class_name = op_name_to_class(args.op_name)
    try:
        import importlib

        mod = importlib.import_module("ezops")
        getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        print(f"Error: cannot import {class_name} from ezops: {e}")
        sys.exit(1)

    # Warm up GPU clocks
    print(f"Warming up GPU ({args.warmup} iters, {args.op_name}) ...")
    try:
        warmup(args.op_name, pos_args, kw_args, args.warmup)
    except Exception as e:
        print(f"Warning: warmup failed: {e}")

    # Profile each kernel
    for kernel_name in kernels:
        label = f"{args.op_name}/{kernel_name}"
        print(f"\nProfiling {label} ...")

        script = generate_profile_script(args.op_name, kernel_name, pos_args, kw_args)
        script_path = output_dir / f"_run_{args.op_name}_{kernel_name}.py"
        script_path.write_text(script)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{args.op_name}_{kernel_name}_{timestamp}"

        cmd = [
            ncu_path,
            "--set",
            "full",
            "-o",
            str(output_file),
            "--force-overwrite",
            sys.executable,
            str(script_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})")
            if result.stderr:
                # Show last few lines of stderr for diagnosis
                tail = result.stderr.strip().split("\n")
                for line in tail[-5:]:
                    print(f"  {line}")
        else:
            ncu_file = output_dir / f"{args.op_name}_{kernel_name}_{timestamp}.ncu-rep"
            if ncu_file.exists():
                size_kb = ncu_file.stat().st_size / 1024
                print(f"  Saved: {ncu_file} ({size_kb:.1f} KB)")
            else:
                print(f"  Done (check {output_dir}/ for output files)")

        script_path.unlink(missing_ok=True)

    print(f"\nDone. Reports in {output_dir}/")


if __name__ == "__main__":
    main()
