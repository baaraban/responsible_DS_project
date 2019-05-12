import shap
from internal_scripts.descriptors.BaseDescriptor import BaseDescriptor


class ShapDescriptor(BaseDescriptor):
    def __init__(self):
        super().__init__()

    def get_descriptor_name(self):
        return "Shap"

    def get_explainer(self, model_name, model, data_dict):
        kernel_models = 'SVC'
        deep_models = 'Keras'
        if kernel_models in model_name:
            return shap.KernelExplainer(model.predict_proba, data_dict['x_train'])
        if deep_models in model_name:
            return shap.DeepExplainer(model, data_dict['x_train'])
        return shap.TreeExplainer(model)

    def describe(self, model_name, model, data_dict, n=300):
        try:
            explainer = self.get_explainer(model_name, model, data_dict)
            feature_names = data_dict['x_train'].columns
            shap_values = explainer.shap_values(data_dict['x_train'][:n])
            if type(shap_values) != list:
                return {
                    "Force plot": shap.force_plot(explainer.expected_value, shap_values,
                                                  feature_names=feature_names)
                }
            result_dict = {}
            for i in range(len(shap_values)):
                result_dict[f"Force plot {i} class"] = shap.force_plot(explainer.expected_value[i], shap_values[i],
                                                    feature_names=feature_names)
            return result_dict
        except Exception as e:
            print(e)
            return {}
