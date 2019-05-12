from tensorflow import keras
from tensorflow.keras.layers import Dense
from keras.regularizers import l2


def get_keras_model(data_dict):
    n_cols = data_dict['x_train'].shape[1]
    n_classes = len(data_dict['y_train'].unique())

    model = keras.Sequential([
        Dense(32, activation='relu', input_shape=(n_cols,)),
        Dense(16, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),activation='relu'),
        Dense(8, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'),
        Dense(4, activation='relu'),
        keras.layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
