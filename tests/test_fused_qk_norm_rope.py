import pytest
import torch

from ezops import FusedQkNormRopeOp, list_backends

# qwen3-vl-2B head layout (16 query heads, 8 kv heads, head_dim 128)
QWEN3_VL_2B = dict(num_heads=16, head_dim=128, num_kv_heads=8)


def _run(batch, max_seq_len, position, num_kv_heads):
    kwargs = {**QWEN3_VL_2B, "num_kv_heads": num_kv_heads}
    torch.manual_seed(42)
    ref_op = FusedQkNormRopeOp(
        batch, backend="ref", max_seq_len=max_seq_len, position=position, **kwargs
    )
    data = ref_op.gen_data()
    # Pristine cache from before any forward: slot `position` holds random
    # scratch, so a backend that skips the KV write-back can't pass by
    # inheriting ref's values.
    K0, V0 = data[7].clone(), data[8].clone()
    out_ref = ref_op._ref_forward(*data)  # writes ref k/v into data's slot
    Kc_ref, Vc_ref = data[7], data[8]

    for backend in list_backends("fused_qk_norm_rope"):
        args = tuple(t.clone() for t in data)
        op = FusedQkNormRopeOp(
            batch, backend=backend, max_seq_len=max_seq_len, position=position, **kwargs
        )
        out = op(*args)
        Kc, Vc = args[7], args[8]

        q, k = out
        q_ref, k_ref = out_ref
        assert q.shape == q_ref.shape and k.shape == k_ref.shape
        max_diff = max((q - q_ref).abs().max().item(), (k - k_ref).abs().max().item())
        assert op.check(out, out_ref), f"{backend}: max_diff={max_diff:.3e} exceeds atol/rtol"
        # The token's rotated k and raw v must land in the cache slot matching
        # ref (both round the same fp32 pipeline to bf16).
        assert torch.allclose(Kc[:, :, position, :], Kc_ref[:, :, position, :], atol=1e-2, rtol=1e-2), (
            f"{backend}: k write-back differs from ref"
        )
        assert torch.equal(Vc[:, :, position, :], Vc_ref[:, :, position, :]), (
            f"{backend}: v write-back differs from ref"
        )
        # all other cache slots must be untouched
        mask = torch.ones(max_seq_len, dtype=torch.bool, device="cuda")
        mask[position] = False
        assert torch.equal(Kc[:, :, mask, :], K0[:, :, mask, :])
        assert torch.equal(Vc[:, :, mask, :], V0[:, :, mask, :])


@pytest.mark.parametrize(
    "batch,max_seq_len,position,num_kv_heads",
    [
        (1, 128, 0, 8),  # first token
        (1, 1024, 512, 8),  # mid-cache position
        (2, 1025, 1024, 8),  # batched, last slot, non power-of-two length
        (1, 4096, 1, 4),  # different GQA group
        (1, 100, 50, 16),  # MHA (kv heads == q heads)
        (1, 33, 17, 2),  # odd cache length
    ],
)
def test_fused_qk_norm_rope_backends(batch, max_seq_len, position, num_kv_heads):
    _run(batch, max_seq_len, position, num_kv_heads)


def test_head_dim_power_of_two_check():
    with pytest.raises(ValueError, match="power of two"):
        FusedQkNormRopeOp(1, num_heads=16, head_dim=100, max_seq_len=128, position=0)


def test_position_range_check():
    with pytest.raises(ValueError, match="position"):
        FusedQkNormRopeOp(1, num_heads=16, head_dim=128, max_seq_len=128, position=128)


def test_gqa_divisibility_check():
    with pytest.raises(ValueError, match="divisible"):
        FusedQkNormRopeOp(1, num_heads=16, head_dim=128, max_seq_len=128, position=0, num_kv_heads=6)
