from abc import ABC, abstractmethod


class Op(ABC):
    _backend: str

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    @abstractmethod
    def _ref_forward(self, *args, **kwargs): ...

    @abstractmethod
    def gen_data(self): ...

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
