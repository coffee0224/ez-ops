_KERNEL_REGISTRY: dict[tuple[str, str], type] = {}


def register_kernel(op_name: str, backend: str):
    def decorator(cls):
        # Deferred import: tl_cache imports tilelang, and kernel modules import this
        # registry at module load, so a top-level import would add a heavy dependency
        # (and an import cycle) to CUDA/Triton-only paths.
        from .kernels.tl_cache import kernel_name_scope

        key = (op_name, backend)
        if key in _KERNEL_REGISTRY:
            raise ValueError(
                f"Kernel already registered for ({op_name!r}, {backend!r}): "
                f"{_KERNEL_REGISTRY[key].__name__}"
            )
        cls.op_name = op_name
        cls.backend = backend

        # TileLang compiles lazily inside the first __call__; running calls under the
        # naming scope is what puts the cache under .tilelang/<op>_<backend>_<hash>.
        # For CUDA/Triton kernels the scope is inert. Nested registered-kernel calls
        # resolve to the innermost name via the contextvar reset.
        original_call = cls.__call__

        def call_with_kernel_name(self, *args, **kwargs):
            with kernel_name_scope(op_name, backend):
                return original_call(self, *args, **kwargs)

        cls.__call__ = call_with_kernel_name

        _KERNEL_REGISTRY[key] = cls
        return cls

    return decorator


def get_kernel(op_name: str, backend: str) -> type:
    key = (op_name, backend)
    if key not in _KERNEL_REGISTRY:
        available = [b for (n, b) in _KERNEL_REGISTRY if n == op_name]
        raise KeyError(
            f"No kernel registered for ({op_name!r}, {backend!r}). "
            f"Available backends for {op_name!r}: {available!r}"
        )
    return _KERNEL_REGISTRY[key]


def list_backends(op_name: str) -> list[str]:
    return [b for (n, b) in _KERNEL_REGISTRY if n == op_name]


def list_ops() -> list[str]:
    return sorted({n for (n, _) in _KERNEL_REGISTRY})
