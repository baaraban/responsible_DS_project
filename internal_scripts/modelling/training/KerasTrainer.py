from sklearn.metrics import classification_report

from internal_scripts.modelling.training.abstractions.ModelTrainerBase import ModelTrainerBase


class KerasTrainer(ModelTrainerBase):
    def __init__(self, model_name, model, data_dict):
        super().__init__(model_name, model, data_dict)
        self._SAVING_ROOT = f'{self._SAVING_ROOT}keras/'

    def tune(self, tune_parameters):
        pass

    def _prepare_data(self):
        import pandas as pd
        tr_data = {}

        tr_data['x_train'] = self.data_dict['x_train'].values
        y_train_df = pd.get_dummies(self.data_dict['y_train'], prefix='category')
        tr_data['y_train'] = y_train_df.values

        tr_data['x_test'] = self.data_dict['x_test'].values
        y_test_df = pd.get_dummies(self.data_dict['y_test'], prefix='category')
        tr_data['y_test'] = y_test_df.values
        return tr_data

    def fit(self, fitting_params):
        training_data = self._prepare_data()
        self.model.fit(training_data['x_train'],
                       training_data['y_train'],
                       batch_size=fitting_params['batch_size'],
                       epochs=fitting_params['epochs'],
                       validation_data=(training_data['x_test'], training_data['y_test']))
        return self.model

    def save_model(self):
        filename = f'{self._SAVING_ROOT}{self.model_name}.h5'
        self.model.save(filename)

    def score_model(self):
        y_test_prediction = self.model.predict_classes(self.data_dict['x_test'])
        return classification_report(self.data_dict['y_test'], y_test_prediction)
