import pytest
import torch

from ezops import Qwen3DenseDecodeOp, list_backends

# qwen3-vl-2B per-layer shape (hidden 2048, intermediate 6144, head_dim 128,
# 16 query heads, 8 kv heads)
QWEN3_VL_2B = dict(
    hidden_size=2048,
    intermediate_size=6144,
    num_heads=16,
    head_dim=128,
    num_kv_heads=8,
)


def _run(batch, seq_len, num_kv_heads):
    kwargs = {**QWEN3_VL_2B, "num_kv_heads": num_kv_heads}
    torch.manual_seed(42)
    ref_op = Qwen3DenseDecodeOp(batch, seq_len, backend="ref", **kwargs)
    data = ref_op.gen_data()
    # Pristine cache from before any forward: the last slot holds random
    # scratch, so a backend that skips the KV write-back can't pass by
    # inheriting ref's values.
    K0, V0 = data[1].clone(), data[2].clone()
    out_ref = ref_op._ref_forward(*data)  # writes ref k/v into data's last slot
    Kc_ref, Vc_ref = data[1], data[2]

    for backend in list_backends("qwen3_dense_decode"):
        args = tuple(t.clone() for t in data)
        args = (args[0], K0.clone(), V0.clone()) + args[3:]
        op = Qwen3DenseDecodeOp(batch, seq_len, backend=backend, **kwargs)
        out = op(*args)
        Kc, Vc = args[1], args[2]

        assert out.shape == out_ref.shape
        max_diff = (out - out_ref).abs().max().item()
        assert op.check(out, out_ref), f"{backend}: max_diff={max_diff:.3e} exceeds atol/rtol"
        # The new token's k/v must land in the cache's last slot, matching ref.
        # Not bit-exact: ref rounds qkv to bf16 before the head-norm, backends
        # may norm the fp32 accumulator — both are valid bf16 stagings.
        assert torch.allclose(Kc[:, :, -1, :], Kc_ref[:, :, -1, :], atol=1e-2, rtol=1e-2), (
            f"{backend}: k write-back differs from ref"
        )
        assert torch.allclose(Vc[:, :, -1, :], Vc_ref[:, :, -1, :], atol=1e-2, rtol=1e-2), (
            f"{backend}: v write-back differs from ref"
        )
        # earlier cache slots must be untouched
        assert torch.equal(Kc[:, :, :-1, :], K0[:, :, :-1, :])
        assert torch.equal(Vc[:, :, :-1, :], V0[:, :, :-1, :])


@pytest.mark.parametrize(
    "batch,seq_len,num_kv_heads",
    [
        (1, 128, 8),  # aligned cache
        (1, 1000, 8),  # non power-of-two cache length
        (2, 1025, 8),  # batched, odd length
        (1, 4096, 4),  # different GQA group
        (1, 1, 8),  # cache holds only the token being decoded
    ],
)
def test_qwen3_dense_decode_backends(batch, seq_len, num_kv_heads):
    _run(batch, seq_len, num_kv_heads)


def test_gqa_divisibility_check():
    with pytest.raises(ValueError, match="divisible"):
        Qwen3DenseDecodeOp(1, 128, num_heads=16, head_dim=128, hidden_size=2048, intermediate_size=6144, num_kv_heads=6)
