"""Keep TileLang's compile cache inside the repo, named after the registered kernel.

TileLang's default layout is ``~/.tilelang/cache/<version>/<platform>/kernels/<sha256>``,
so the generated ``device_kernel.cu`` of every kernel is buried under an unreadable
hash. ezops instead wants:

    <repo>/.tilelang/<op_name>_<backend>_<hash>/device_kernel.cu

``setup()`` redirects the whole TileLang cache root (kernel dirs plus tmp/staging/
cuda-binaries) into the repo, and patches ``KernelCache._get_cache_path`` so that
kernel entries created while a ``kernel_name_scope`` is active get the
``<op>_<backend>_<hash>`` directory name. The hash stays TileLang's own cache key,
so cache invalidation is unchanged: touching the kernel source produces a new hash.

Setting ``TILELANG_CACHE_DIR`` in the environment keeps working; it only moves the
root, the naming scheme still applies.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from tilelang import env as _tl_env
from tilelang.cache.kernel_cache import KernelCache

REPO_ROOT = Path(__file__).resolve().parents[2]

# (op_name, backend) of the registered kernel currently executing; compile happens
# lazily inside the first __call__, so the scope only needs to cover kernel calls.
_active_kernel: ContextVar[tuple[str, str] | None] = ContextVar(
    "ezops_tilelang_kernel", default=None
)

_original_get_cache_path = KernelCache._get_cache_path

CACHE_ROOT = REPO_ROOT / ".tilelang"


def _named_get_cache_path(self, key: str) -> str:
    kernel = _active_kernel.get()
    if kernel is None:
        return _original_get_cache_path(self, key)
    op_name, backend = kernel
    return os.path.join(str(CACHE_ROOT), f"{op_name}_{backend}_{key}")


def setup(root: str | os.PathLike | None = None) -> Path:
    """Redirect the TileLang cache; call once at import of ezops.kernels."""
    global CACHE_ROOT
    if root is not None:
        CACHE_ROOT = Path(root)
    elif "TILELANG_CACHE_DIR" in os.environ:
        CACHE_ROOT = Path(os.environ["TILELANG_CACHE_DIR"])
    else:
        CACHE_ROOT = REPO_ROOT / ".tilelang"

    # Environment is the tilelang.env.Environment instance; assigning goes through
    # EnvVar.__set__ (forced override, re-read on every access), so this also
    # moves tmp/, staging/ and cuda-binaries even if KernelCache already ran.
    _tl_env.TILELANG_CACHE_DIR = str(CACHE_ROOT)

    if KernelCache._get_cache_path is not _named_get_cache_path:
        KernelCache._get_cache_path = _named_get_cache_path

    os.makedirs(CACHE_ROOT, exist_ok=True)
    return CACHE_ROOT


@contextmanager
def kernel_name_scope(op_name: str, backend: str):
    """Compile every TileLang kernel inside this scope into <op>_<backend>_<hash>."""
    token = _active_kernel.set((op_name, backend))
    try:
        yield
    finally:
        _active_kernel.reset(token)
