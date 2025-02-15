from abc import ABC, abstractmethod


class ModelTrainerBase(ABC):
    def __init__(self, model_name, model, data_dict):
        self._SAVING_ROOT = 'models/'
        self.model_name = model_name
        self.model = model
        self.data_dict = data_dict
        super().__init__()

    @abstractmethod
    def tune(self, tune_parameters):
        pass

    @abstractmethod
    def fit(self, data_dict, fitting_params):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def score_model(self):
        pass
