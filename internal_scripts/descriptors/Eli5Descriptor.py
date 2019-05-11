import eli5
from internal_scripts.descriptors.BaseDescriptor import BaseDescriptor


class Eli5Descriptor(BaseDescriptor):
    def __init__(self):
        super().__init__()

    def describe(self, model_name, model, data_dict):
        feature_names = list(data_dict['x_train'].columns)
        test_observation = data_dict['x_test'].iloc[0]
        explained_weights = eli5.show_weights(model, feature_names=feature_names)
        explained_prediction = eli5.show_prediction(model, test_observation)

        return {
            "Weights explanation": explained_weights,
            "Predictions explanation": explained_prediction
        }
