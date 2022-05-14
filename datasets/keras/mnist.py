import numpy as np
from tensorflow import keras
from datasets.keras.base_keras_dataset import BaseKerasDataSet

class MnistDataSet(BaseKerasDataSet):
    def __init__(self):
        super().__init__()
        print("Initializing MNIST Dataset")
        self.num_classes = 10
        self.input_shape = (28, 28, 1)

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        # Make sure images have shape (28, 28, 1)
        self.x_train = np.expand_dims(x_train, -1)
        self.x_test = np.expand_dims(x_test, -1)

        self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.num_classes)

        print("x_train shape:", self.x_train.shape)
        print(self.x_train.shape[0], "train samples")
        print(self.x_test.shape[0], "test samples")