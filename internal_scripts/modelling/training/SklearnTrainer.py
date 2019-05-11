from datetime import datetime
import pickle

from internal_scripts.modelling.training.abstractions.ModelTrainerBase import ModelTrainerBase


class SklearnTrainer(ModelTrainerBase):
    def __init__(self, model_name, model):
        super().__init__(model_name, model)
        self._SAVING_ROOT = f'{super().__SAVING_ROOT}sklearn/'

    def tune(self, tune_parameters):
        pass

    def fit(self, data_dict, fitting_params):
        self.model.fit(data_dict['x_train'], data_dict['y_train'])
        return self.model

    def save_model(self):
        filename = f'{self._SAVING_ROOT}{self.model_name}/{str(datetime.utcnow())}.sav'
        pickle.dump(self.model, open(filename, 'wb'))
