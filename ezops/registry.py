_KERNEL_REGISTRY: dict[tuple[str, str], type] = {}


def register_kernel(op_name: str, backend: str):
    def decorator(cls):
        key = (op_name, backend)
        if key in _KERNEL_REGISTRY:
            raise ValueError(
                f"Kernel already registered for ({op_name!r}, {backend!r}): "
                f"{_KERNEL_REGISTRY[key].__name__}"
            )
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
