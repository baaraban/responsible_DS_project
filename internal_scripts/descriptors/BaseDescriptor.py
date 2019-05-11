from abc import ABC, abstractmethod


class BaseDescriptor(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def describe(self, model_name, model, params):
        pass
