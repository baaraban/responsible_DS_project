def get_saved_models(dataset_name):
    import os
    import pickle
    from keras.models import load_model

    def get_sklearn_models(path):
        sklearn_model_ending = '.sav'
        return {f.split('.')[0]: pickle.load(open(f'{path}{f}', 'rb')) for f in os.listdir(path) if
                f.endswith(sklearn_model_ending) and dataset_name in f}

    def get_keras_models(path):
        keras_model_ending = '.h5'
        return {f.split('.')[0]: load_model(f'{path}{f}') for f in os.listdir(path) if
                f.endswith(keras_model_ending) and dataset_name in f}

    path_to_models = 'models/'
    path_to_sklearn = f'{path_to_models}sklearn/'
    path_to_keras = f'{path_to_models}keras/'

    return dict(get_sklearn_models(path_to_sklearn),
                **get_keras_models(path_to_keras))
