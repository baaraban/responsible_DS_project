from abc import ABC, abstractmethod


class BaseDescriptor(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_descriptor_name(self):
        pass

    @abstractmethod
    def describe(self, model_name, model, data_dict):
        pass
