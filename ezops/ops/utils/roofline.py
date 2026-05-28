from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.flop_counter import FlopCounterMode


def _elemwise_flop(*input_shapes, out_shape=None, **kwargs) -> int:
    if out_shape is not None:
        n = 1
        for d in out_shape:
            n *= d
        return n
    return 0


_ELEMENTWISE_OPS = [
    torch.ops.aten.add,
    torch.ops.aten.sub,
    torch.ops.aten.mul,
    torch.ops.aten.div,
    torch.ops.aten.neg,
    torch.ops.aten.abs,
    torch.ops.aten.exp,
    torch.ops.aten.log,
    torch.ops.aten.sqrt,
    torch.ops.aten.rsqrt,
    torch.ops.aten.relu,
    torch.ops.aten.sigmoid,
    torch.ops.aten.tanh,
    torch.ops.aten.silu,
    torch.ops.aten.gelu,
    torch.ops.aten.pow,
    torch.ops.aten.copy,
    torch.ops.aten.clone,
    torch.ops.aten.where,
    torch.ops.aten.maximum,
    torch.ops.aten.minimum,
]

def _mv_flop(*input_shapes, out_shape=None, **kwargs) -> int:
    if len(input_shapes) >= 2 and len(input_shapes[0]) == 2:
        return 2 * input_shapes[0][0] * input_shapes[0][1]
    return 0


_FLOP_REGISTRY = {
    **{op: _elemwise_flop for op in _ELEMENTWISE_OPS},
    torch.ops.aten.mv: _mv_flop,
}


@dataclass(frozen=True)
class RooflineResult:
    flops: int
    bytes: int
    fused_bytes: int
    arithmetic_intensity: float = 0.0

    def __post_init__(self):
        if self.bytes > 0:
            object.__setattr__(self, "arithmetic_intensity", self.flops / self.bytes)
        else:
            object.__setattr__(self, "arithmetic_intensity", float("inf"))


def _output_nbytes(result) -> int:
    if result is None:
        return 0
    if isinstance(result, torch.Tensor):
        return result.nbytes
    if isinstance(result, (tuple, list)):
        return sum(t.nbytes for t in result if isinstance(t, torch.Tensor))
    return 0


def measure_roofline(fn, args: tuple, kwargs: dict) -> RooflineResult:
    with FlopCounterMode(display=False, custom_mapping=_FLOP_REGISTRY) as counter:
        result = fn(*args, **kwargs)

    flops = counter.get_total_flops()

    total_bytes = sum(a.nbytes for a in args if isinstance(a, torch.Tensor))
    total_bytes += sum(v.nbytes for v in kwargs.values() if isinstance(v, torch.Tensor))
    total_bytes += _output_nbytes(result)

    return RooflineResult(flops=flops, bytes=total_bytes, fused_bytes=total_bytes)
