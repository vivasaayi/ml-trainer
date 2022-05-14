import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# https://keras.io/examples/vision/mnist_convnet/

class SimpleConvnet():
    def initialize(self, params):
        print(params)
        self.model = keras.Sequential(
            [
                keras.Input(shape=params["input_shape"]),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(params["num_classes"], activation="softmax"),
            ]
        )