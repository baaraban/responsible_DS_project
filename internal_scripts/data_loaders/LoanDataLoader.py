from internal_scripts.data_loaders.DataLoaderBase import DataLoaderBase

import pandas as pd
from sklearn.model_selection import train_test_split


class LoanDataLoader(DataLoaderBase):
    __LOAN_PATH = 'datasets/preprocessed/pre_loan_data.csv'

    def __init__(self):
        super().__init__()

    def get_train_test_split(self, test_size=.3, rand_state=None):
        df = pd.read_csv(self.__LOAN_PATH)

        x = df.drop(['Loan Status'], axis=1)
        y = df['Loan Status']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=rand_state)

        return {
            "dataset_name": "Loan_Data",
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test
        }