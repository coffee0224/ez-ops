import pytest
import torch

from ezops import SoftmaxOp, list_backends
from ezops.ops.utils.accuracy import check_determinism, check_input_readonly


def _run(batch, dim):
    torch.manual_seed(42)
    ref_op = SoftmaxOp(batch_size=batch, dim=dim, backend="ref")
    X, Out_ref = ref_op.gen_data()
    ref_op._ref_forward(X, Out_ref)

    for backend in list_backends("softmax"):
        op = SoftmaxOp(batch_size=batch, dim=dim, backend=backend)
        Out = torch.empty_like(Out_ref)
        op(X, Out)
        max_diff = (Out - Out_ref).abs().max().item()
        assert op.check(Out, Out_ref), f"{backend}: max_diff={max_diff:.3e} exceeds atol/rtol"
        assert check_determinism(op, inputs=(X,), outputs=Out), f"{backend}: nondeterministic output"
        assert check_input_readonly(op, inputs=(X,), outputs=Out), f"{backend}: mutates input"


@pytest.mark.parametrize(
    "batch,dim",
    [
        (1, 64),  # single row, dim < block size
        (128, 1024),  # typical head/logit dim
        (3, 100),  # non-power-of-two dim
        (7, 4097),  # dim straddles the chunk boundary
        (4096, 4096),  # large batch
    ],
)
def test_softmax_backends(batch, dim):
    _run(batch, dim)
