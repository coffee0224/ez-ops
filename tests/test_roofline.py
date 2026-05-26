import pytest
import torch

from ezops import VectorAddOp
from ezops.ops.utils.roofline import RooflineResult, measure_roofline


# ---------------------------------------------------------------------------
# VectorAddOp roofline
# ---------------------------------------------------------------------------


class TestVectorAddRoofline:
    """VectorAdd: C.copy_(A + B)  ->  flops = N, bytes = 3*N*sizeof(float32)."""

    @pytest.mark.parametrize("n", [1, 64, 256, 1024, 1 << 20])
    def test_flops(self, n):
        op = VectorAddOp(n=n, backend="triton")
        r = op.get_roofline()
        assert r.flops == n

    @pytest.mark.parametrize("n", [1, 64, 256, 1024, 1 << 20])
    def test_bytes(self, n):
        op = VectorAddOp(n=n, backend="triton")
        r = op.get_roofline()
        expected_bytes = 3 * n * 4  # A + B + C, float32
        assert r.bytes == expected_bytes

    @pytest.mark.parametrize("n", [1, 64, 256, 1024])
    def test_fused_equals_bytes(self, n):
        op = VectorAddOp(n=n, backend="triton")
        r = op.get_roofline()
        assert r.fused_bytes == r.bytes

    @pytest.mark.parametrize("n", [64, 1024, 1 << 20])
    def test_arithmetic_intensity(self, n):
        op = VectorAddOp(n=n, backend="triton")
        r = op.get_roofline()
        expected_ai = 1 / 12  # N / (3 * N * 4)
        assert abs(r.arithmetic_intensity - expected_ai) < 1e-10


# ---------------------------------------------------------------------------
# measure_roofline with raw functions (return-value style)
# ---------------------------------------------------------------------------


class TestMeasureRoofline:
    def test_elementwise_add(self):
        n = 1024

        def fn(a, b):
            return a + b

        a = torch.randn(n)
        b = torch.randn(n)
        r = measure_roofline(fn, (a, b), {})
        assert r.flops == n
        assert r.bytes == a.nbytes + b.nbytes + n * 4  # 2 reads + 1 write

    def test_matmul(self):
        m, k, n = 32, 64, 128

        def fn(a, b):
            return a @ b

        a = torch.randn(m, k)
        b = torch.randn(k, n)
        r = measure_roofline(fn, (a, b), {})
        assert r.flops == 2 * m * k * n
        assert r.bytes == a.nbytes + b.nbytes + m * n * 4

    def test_chain_elementwise(self):
        n = 512

        def fn(a, b):
            return torch.relu(a * b + a)

        a = torch.randn(n)
        b = torch.randn(n)
        r = measure_roofline(fn, (a, b), {})
        # mul + add + relu = 3 * n
        assert r.flops == 3 * n
        assert r.bytes == a.nbytes + b.nbytes + n * 4

    def test_inplace_output_pattern(self):
        n = 256

        def fn(a, b, c):
            c.copy_(a + b)

        a = torch.randn(n)
        b = torch.randn(n)
        c = torch.empty(n)
        r = measure_roofline(fn, (a, b, c), {})
        assert r.flops == n
        assert r.bytes == 3 * n * 4

    def test_no_tensors_zero_bytes(self):
        def fn(x):
            return x + 1

        r = measure_roofline(fn, (5,), {})
        assert r.flops == 0
        assert r.bytes == 0
        assert r.arithmetic_intensity == float("inf")

    def test_tuple_return(self):
        n = 64

        def fn(a):
            return a * 2, a + 3

        a = torch.randn(n)
        r = measure_roofline(fn, (a,), {})
        assert r.flops == 2 * n
        assert r.bytes == a.nbytes + 2 * n * 4


# ---------------------------------------------------------------------------
# RooflineResult dataclass
# ---------------------------------------------------------------------------


class TestRooflineResult:
    def test_frozen(self):
        r = RooflineResult(flops=100, bytes=200, fused_bytes=200)
        with pytest.raises(AttributeError):
            r.flops = 0

    def test_arithmetic_intensity_auto(self):
        r = RooflineResult(flops=100, bytes=400, fused_bytes=400)
        assert r.arithmetic_intensity == 0.25

    def test_zero_bytes_infinite_ai(self):
        r = RooflineResult(flops=100, bytes=0, fused_bytes=0)
        assert r.arithmetic_intensity == float("inf")
