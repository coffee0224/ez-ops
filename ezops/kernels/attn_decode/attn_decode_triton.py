import torch
import triton
import triton.language as tl

from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("attn_decode", "triton")
class AttnDecodeTritonKernel(BaseKernel):
    def __init__(self, batch: int, num_heads: int, seq_len: int, head_dim: int):
        self.batch = batch
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.block_size = 1024  # TODO: tune for this op

    @staticmethod
    @triton.jit
    def _kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        batch, num_heads, seq_len, head_dim,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_km, stride_kd,
        stride_vb, stride_vh, stride_vm, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        BLOCK_SIZE: tl.constexpr,
    ):
        # TODO: implement the triton kernel for attn_decode
        raise NotImplementedError("TODO: implement triton kernel for attn_decode")

    def __call__(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.shape == (self.batch, self.num_heads, 1, self.head_dim)
        assert K.shape == (self.batch, self.num_heads, self.seq_len, self.head_dim)
        assert V.shape == (self.batch, self.num_heads, self.seq_len, self.head_dim)

        out = torch.empty(self.batch, self.num_heads, 1, self.head_dim, device=Q.device, dtype=torch.bfloat16)
        grid = lambda meta: (self.batch * self.num_heads,)
        self._kernel[grid](
            Q, K, V, out,
            self.batch, self.num_heads, self.seq_len, self.head_dim,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_SIZE=self.block_size,
        )
        return out
