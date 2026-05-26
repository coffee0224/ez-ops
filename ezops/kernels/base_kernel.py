from abc import ABC, abstractmethod


class BaseKernel(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs): ...
