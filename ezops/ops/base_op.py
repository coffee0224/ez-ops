from abc import ABC, abstractmethod

import torch

from .utils.roofline import RooflineResult, measure_roofline


class Op(ABC):
    _backend: str
    _atol: float = 1e-6
    _rtol: float = 1e-5
    _params_desc: dict[str, str] = {}

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    @abstractmethod
    def _ref_forward(self, *args, **kwargs): ...

    @abstractmethod
    def gen_data(self): ...

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_roofline(self) -> RooflineResult:
        data = self.gen_data()
        if not isinstance(data, tuple):
            data = (data,)
        return measure_roofline(self._ref_forward, data, {})

    def check(self, actual, expected) -> bool:
        return torch.allclose(actual, expected, atol=self._atol, rtol=self._rtol)
