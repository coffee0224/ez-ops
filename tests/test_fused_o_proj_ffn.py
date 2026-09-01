import pytest
import torch

from ezops import FusedOProjFfnOp, list_backends

# qwen3-vl-2B per-layer shape (hidden 2048, intermediate 6144, head_dim 128,
# 16 query heads -> 2048 attention lanes)
QWEN3_VL_2B = dict(hidden_size=2048, intermediate_size=6144, num_heads=16, head_dim=128)

# qwen3-0.6B: attention lanes (2048) are wider than the hidden (1024)
QWEN3_0_6B = dict(hidden_size=1024, intermediate_size=3072, num_heads=16, head_dim=128)


def _run(batch, **kwargs):
    torch.manual_seed(42)
    ref_op = FusedOProjFfnOp(batch, backend="ref", **kwargs)
    data = ref_op.gen_data()
    out_ref = ref_op._ref_forward(*data)

    for backend in list_backends("fused_o_proj_ffn"):
        op = FusedOProjFfnOp(batch, backend=backend, **kwargs)
        out = op(*data)

        assert out.shape == out_ref.shape
        max_diff = (out - out_ref).abs().max().item()
        assert op.check(out, out_ref), f"{backend}: max_diff={max_diff:.3e} exceeds atol/rtol"


@pytest.mark.parametrize(
    "batch,kwargs",
    [
        (1, QWEN3_VL_2B),
        (1, QWEN3_0_6B),
        (2, QWEN3_VL_2B),
        (4, QWEN3_VL_2B),
    ],
)
def test_fused_o_proj_ffn_backends(batch, kwargs):
    _run(batch, **kwargs)


def test_shape_divisibility_check():
    # tile-divisibility is enforced by the backend kernel, not the ref path
    with pytest.raises(ValueError, match="divisible"):
        FusedOProjFfnOp(1, hidden_size=2048, intermediate_size=6200, num_heads=16, head_dim=128, backend="triton")
