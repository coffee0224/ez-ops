from ..base_kernel import BaseKernel
from ...registry import register_kernel


@register_kernel("attn_decode", "tilelang")
class AttnDecodeTileLangKernel(BaseKernel):
    def __init__(self, batch: int, num_heads: int, seq_len: int, head_dim: int):
        self.batch = batch
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.block_size = 1024  # TODO: tune for this op
        self._program = self._build()

    def _build(self):
        # TODO: implement the tilelang kernel for attn_decode
        raise NotImplementedError("TODO: implement tilelang kernel for attn_decode")

    def __call__(self, Q, K, V):
        # TODO: implement the call logic for attn_decode
        raise NotImplementedError("TODO: implement tilelang __call__ for attn_decode")
