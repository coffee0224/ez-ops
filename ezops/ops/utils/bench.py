from __future__ import annotations

from typing import Any, Callable, Optional

import torch

# ---------------------------------------------------------------------------
# L2 cache flush buffer (sized to actual L2, allocated lazily)
# ---------------------------------------------------------------------------

_l2_flush_cache: Optional[torch.Tensor] = None


def _get_l2_flush_cache() -> torch.Tensor:
    global _l2_flush_cache
    if _l2_flush_cache is None:
        l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
        if l2_bytes <= 0:
            l2_bytes = int(256e6)
        _l2_flush_cache = torch.empty(l2_bytes // 4, dtype=torch.int, device="cuda")
    return _l2_flush_cache


def bench_kernel(
    fn: Callable,
    args: tuple[Any, ...] = (),
    n_warmup: int = 10,
    n_repeat: int = 50,
    n_trials: int = 3,
) -> float:
    """Benchmark a GPU kernel with L2 flush and multi-trial median.

    Uses CUDA Graph capture (when possible) to eliminate CPU dispatch
    overhead, so measured latency reflects GPU kernel execution only.

    Adapted from NVIDIA SOL-ExecBench protocol:
      1. Run n_warmup iterations with L2 flush.
      2. Capture a CUDA Graph of the kernel (falls back to eager if
         the function is not graph-capturable).
      3. For each of n_trials, run n_repeat iterations with L2 flush
         and CUDA event timing.
      4. Return the median trial mean (robust to outliers).

    Args:
        fn: Callable to benchmark. Called as ``fn(*args)``.
        args: Tensor arguments to pass through.
        n_warmup: Warmup iterations.
        n_repeat: Timed iterations per trial.
        n_trials: Independent trials.

    Returns:
        Kernel latency in milliseconds.
    """
    cache = _get_l2_flush_cache()
    has_args = len(args) > 0

    # Pre-clone a small pool of input tensors so the kernel sees different
    # addresses across iterations. Skip cloning if total tensor memory
    # exceeds 1 GB to avoid OOM on large workloads.
    _N_CLONES = 3
    _MAX_CLONE_BYTES = 1 << 30
    if has_args:
        tensor_mask = tuple(isinstance(a, torch.Tensor) for a in args)
        total_bytes = sum(
            a.nelement() * a.element_size()
            for a, m in zip(args, tensor_mask, strict=True)
            if m
        )
        if total_bytes * _N_CLONES <= _MAX_CLONE_BYTES:
            arg_pool = [
                tuple(a.clone() if m else a for a, m in zip(args, tensor_mask, strict=True))
                for _ in range(_N_CLONES)
            ]

            def _run(i):
                return fn(*arg_pool[i % _N_CLONES])
        else:
            arg_pool = None

            def _run(i):
                return fn(*args)
    else:
        arg_pool = None

        def _run(i):
            return fn()

    # Warmup (no timing)
    for i in range(n_warmup):
        cache.zero_()
        _run(i % n_repeat)
    torch.cuda.synchronize()

    # Try CUDA Graph capture to eliminate CPU dispatch overhead.
    # Falls back to eager execution if the function is not capturable.
    graph = None
    try:
        g = torch.cuda.CUDAGraph()
        if has_args:
            static_args = tuple(
                a.clone() if isinstance(a, torch.Tensor) else a for a in args
            )
            with torch.cuda.graph(g):
                fn(*static_args)
        else:
            with torch.cuda.graph(g):
                fn()
        torch.cuda.synchronize()
        graph = g
    except Exception:
        graph = None

    # Warmup graph replay separately — eager warmup above does not
    # exercise the graph launch path, and the first few replays can be
    # slower due to driver-side graph initialization.
    if graph is not None:
        for _ in range(n_warmup):
            cache.zero_()
            graph.replay()
        torch.cuda.synchronize()

    # Timed trials with CUDA events
    trial_means: list[float] = []
    for _ in range(n_trials):
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
        for i in range(n_repeat):
            cache.zero_()
            start_events[i].record()
            if graph is not None:
                graph.replay()
            else:
                _run(i)
            end_events[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events, strict=True)]
        trial_means.append(sum(times) / len(times))

    # Free arg pool
    if arg_pool is not None:
        del arg_pool
    torch.cuda.empty_cache()

    trial_means.sort()
    return trial_means[len(trial_means) // 2]
