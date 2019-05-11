def get_saved_models():
    import os
    import pickle

    def get_sklearn_models(path):
        sklearn_model_ending = '.sav'
        return {f.split('.')[0]: pickle.load(open(f'{path}{f}', 'rb')) for f in os.listdir(path) if
                f.endswith(sklearn_model_ending)}

    path_to_models = 'models/'
    path_to_sklearn = f'{path_to_models}sklearn/'
    return get_sklearn_models(path_to_sklearn)
