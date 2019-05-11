from internal_scripts.modelling.training.abstractions.ModelTrainerBase import ModelTrainerBase


class KerasTrainer(ModelTrainerBase):
    def tune(self, tune_parameters):
        pass

    def fit(self, data_dict, fitting_params):
        pass

    def save_model(self):
        pass