from datetime import datetime
import pickle
import os

from internal_scripts.modelling.training.abstractions.ModelTrainerBase import ModelTrainerBase
from sklearn.metrics import classification_report


class SklearnTrainer(ModelTrainerBase):
    def __init__(self, model_name, model, data_dict):
        super().__init__(model_name, model, data_dict)
        self._SAVING_ROOT = f'{self._SAVING_ROOT}sklearn/'

    def tune(self, tune_parameters):
        pass

    def fit(self, fitting_params):
        self.model.fit(self.data_dict['x_train'], self.data_dict['y_train'])
        return self.model

    def save_model(self):
        folder_path = f'{self._SAVING_ROOT}{self.model_name}'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        date_string = datetime.now().strftime("%Y_%m_%d__%H_%M")

        filename = f'{folder_path}/{date_string}.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def score_model(self):
        y_test_prediction = self.model.predict(self.data_dict['x_test'])
        return classification_report(self.data_dict['y_test'], y_test_prediction)
