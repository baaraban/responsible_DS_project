from internal_scripts.data_loaders.DataLoaderBase import DataLoaderBase

import pandas as pd
from sklearn.model_selection import train_test_split


class BlackFridayDataLoader(DataLoaderBase):
    __BLACK_FRIDAY_PATH = 'datasets/preprocessed/pre_black_friday.csv'

    def __init__(self):
        super().__init__()

    def get_train_test_split(self, test_size=.3, rand_state=None):
        df = pd.read_csv(self.__BLACK_FRIDAY_PATH)

        x = df.drop(['Purchase_x'], axis=1)
        y = df['Purchase_x']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=rand_state)

        return {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test
        }