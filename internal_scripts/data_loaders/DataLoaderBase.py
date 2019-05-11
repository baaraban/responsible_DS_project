from abc import ABC, abstractmethod


class DataLoaderBase(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_train_test_split(self):
        pass
