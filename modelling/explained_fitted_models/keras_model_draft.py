from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout
import pandas as pd
import numpy as np

from internal_scripts.data_loaders.BlackFridayDataLoader import BlackFridayDataLoader

data_loader = BlackFridayDataLoader()
data = data_loader.get_train_test_split()


X_train = data['x_train'].values
y_train_df = pd.get_dummies(data['y_train'], prefix='category')
y_train = y_train_df.values

X_test = data['x_test'].values
y_test_df = pd.get_dummies(data['y_test'], prefix='category')
y_test = y_test_df.values

n_cols = X_train.shape[1]

model = keras.Sequential([
    Dense(32, activation='relu', input_shape=(n_cols,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(8),
    Dense(4),
    Dense(4),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
model.fit(X_train, y_train, batch_size=2000, epochs=1000, validation_data=(X_test, y_test))