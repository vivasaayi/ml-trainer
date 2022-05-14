import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# https://keras.io/examples/vision/mnist_convnet/

class CNN():
    def initialize(self, params):
        print(params)
        self.model = keras.Sequential(
            [
                keras.Input(shape=params["input_shape"]),
                layers.Conv2D(filters=64, padding="same", kernel_size=3, activation='relu'),
                layers.Conv2D(filters=32, padding="same", kernel_size=3, activation='relu'),
                layers.MaxPool2D(pool_size=2, strides=2),
                layers.Conv2D(filters=16, padding="same", kernel_size=3, activation='relu'),
                layers.Conv2D(filters=16, padding="same", kernel_size=3, activation='relu'),
                layers.MaxPool2D(pool_size=2, strides=2),

                layers.Flatten(),
                layers.Dense(units=128, activation='relu'),
                layers.Dense(units=128, activation='relu'),

                layers.Dense(params["num_classes"], activation="softmax")
            ]
        )