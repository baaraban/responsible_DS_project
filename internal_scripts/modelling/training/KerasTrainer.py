from internal_scripts.modelling.training.abstractions.ModelTrainerBase import ModelTrainerBase


class KerasTrainer(ModelTrainerBase):
    def __init__(self, model_name, model):
        super().__init__(model_name, model)
        self._SAVING_ROOT = f'{self._SAVING_ROOT}keras/'

    def tune(self, tune_parameters):
        pass

    def fit(self, data_dict, fitting_params):
        pass

    def save_model(self):
        pass