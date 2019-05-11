import eli5
from internal_scripts.descriptors.BaseDescriptor import BaseDescriptor


class Eli5Descriptor(BaseDescriptor):
    def __init__(self):
        super().__init__()

    def describe(self, model_name, model, params):
        weight_description = eli5.show_weights(model, feature_names=params['feature_names'])
        prediction_description = eli5.show_prediction(model, params['test_observation'])
        return [weight_description, prediction_description]
