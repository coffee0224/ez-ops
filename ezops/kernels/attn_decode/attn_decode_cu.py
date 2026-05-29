from pathlib import Path

from tvm_ffi import cpp

from ..base_kernel import BaseKernel
from ...registry import register_kernel

_CU_SRC = Path(__file__).with_suffix(".cu")


@register_kernel("attn_decode", "cuda")
class AttnDecodeCudaKernel(BaseKernel):
    def __init__(self, batch: int, num_heads: int, seq_len: int, head_dim: int):
        self.batch = batch
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self._mod = cpp.load_inline(
            name="attn_decode_cuda",
            cuda_sources=_CU_SRC.read_text(),
            functions="attn_decode_cu",
            extra_cuda_cflags=["-O3", "--generate-line-info"],
        )

    def __call__(self, Q, K, V):
        # TODO: implement the cuda call for attn_decode
        raise NotImplementedError("TODO: implement cuda __call__ for attn_decode")
