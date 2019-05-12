import lime
import lime.lime_tabular
from IPython.display import HTML

from internal_scripts.descriptors.BaseDescriptor import BaseDescriptor

class LimeDescriptor(BaseDescriptor):
    def __init__(self):
        super().__init__()

    def get_descriptor_name(self):
        return "Lime"

    def describe(self, model_name, model, data_dict):
        print(model_name)
        if (not 'Decision_Tree' in model_name):
            return {}
        # Note: we currently don't have Lime implementation for this three models
        test_observation = data_dict['x_test'].values[0]

        explainer = lime.lime_tabular.LimeTabularExplainer(
            data_dict['x_train'].values,
            feature_names=data_dict['x_train'].columns.values.tolist(),
            verbose=True, mode='classification')

        exp = explainer.explain_instance(
             test_observation,
             model.predict_proba, num_features=5)

        return {
            "Predictions explanation": HTML(exp.as_html())
        }
