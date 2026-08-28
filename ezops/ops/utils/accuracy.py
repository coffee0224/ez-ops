"""Accuracy checks modeled after the xpuoj judge protocol.

Per testcase the judge runs, in order and all untimed: an SQNR check of
one call against the oracle, a determinism check (two calls on the same
input, outputs compared byte-for-byte), and an input read-only check.
Only then do warmup and timing start. These helpers mirror that locally;
see ~/.agents/skills/xpuoj-optimize/references/judge-mechanics.md.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch

# xpuoj judges typically pass at 28 dB; leave margin when optimizing
# (>= 48 dB is comfortable, single-digit dB means something is broken).
SQNR_THRESHOLD_DB = 28.0


def sqnr_db(ref: torch.Tensor, out: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio in dB.

    SQNR = 10 * log10(||ref||^2 / ||ref - out||^2). Returns inf when the
    outputs match bitwise (zero error power).
    """
    ref = ref.detach().float()
    out = out.detach().float()
    noise = (ref - out).pow(2).sum().item()
    if noise == 0.0:
        return math.inf
    signal = ref.pow(2).sum().item()
    return 10.0 * math.log10(signal / noise)


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Byte-for-byte equality (distinguishes -0.0/+0.0 and NaN payloads)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(_as_bytes(a), _as_bytes(b)))


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def _as_tuple(x) -> tuple:
    if isinstance(x, torch.Tensor):
        return (x,)
    return tuple(x)


def check_determinism(
    fn: Callable,
    inputs: Iterable[torch.Tensor],
    outputs: torch.Tensor | Iterable[torch.Tensor],
) -> bool:
    """Two calls on the same input, outputs compared byte-for-byte."""
    ins = _as_tuple(inputs)
    outs = _as_tuple(outputs)
    fn(*ins, *outs)
    first = [o.clone() for o in outs]
    fn(*ins, *outs)
    return all(bitwise_equal(o, s) for o, s in zip(outs, first, strict=True))


def check_input_readonly(
    fn: Callable,
    inputs: Iterable[torch.Tensor],
    outputs: torch.Tensor | Iterable[torch.Tensor],
) -> bool:
    """Kernel must leave input tensors byte-for-byte untouched."""
    ins = _as_tuple(inputs)
    outs = _as_tuple(outputs)
    before = [i.clone() for i in ins]
    fn(*ins, *outs)
    return all(bitwise_equal(i, b) for i, b in zip(ins, before, strict=True))
