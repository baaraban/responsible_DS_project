from internal_scripts.modelling.training.abstractions.ModelTrainerBase import ModelTrainerBase


class KerasTrainer(ModelTrainerBase):
    def __init__(self, model_name, model):
        super().__init__(model_name, model)
        self._SAVING_ROOT = f'{self._SAVING_ROOT}keras/'

    def tune(self, tune_parameters):
        pass

    def _prepare_data(self, data_dict):
        import pandas as pd
        X_train = data_dict['x_train'].values
        y_train_df = pd.get_dummies(data_dict['y_train'], prefix='category')
        y_train = y_train_df.values

        X_test = data['x_test'].values
        y_test_df = pd.get_dummies(data_dict['y_test'], prefix='category')
        y_test = y_test_df.values
        return None

    def fit(self, data_dict, fitting_params):
        data = self._prepare_data(data_dict)
        pass

    def save_model(self):
        pass

    def score_model(self):
        pass
