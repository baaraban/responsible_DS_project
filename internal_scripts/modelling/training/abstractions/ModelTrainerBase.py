from abc import ABC, abstractmethod


class ModelTrainerBase(ABC):
    def __init__(self, model_name, model):
        self.model_name = model_name
        self.model = model
        super().__init__()

    @abstractmethod
    def tune(self, tune_parameters):
        pass

    @abstractmethod
    def fit(self, data_dict):
        pass

    @abstractmethod
    def save_model(self):
        pass