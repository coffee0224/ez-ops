from .tl_cache import setup as _setup_tilelang_cache

# Must run before any kernel module registers: puts TileLang's compiled kernels in
# <repo>/.tilelang/<op_name>_<backend>_<hash>/ instead of ~/.tilelang/cache/...
_setup_tilelang_cache()

from . import vector_add  # noqa: F401
from . import gemv  # noqa: F401
from . import attn_decode  # noqa: F401
from . import pdl_gemm  # noqa: F401
from . import reduce  # noqa: F401
from . import qwen3_dense_decode  # noqa: F401
