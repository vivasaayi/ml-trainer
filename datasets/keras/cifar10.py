import numpy as np
from tensorflow import keras
from datasets.keras.base_keras_dataset import BaseKerasDataSet

class Cifar10DataSet(BaseKerasDataSet):
    def __init__(self):
        super().__init__()
        print("Initializing CIFAR10 Dataset")
        self.num_classes = 10
        self.input_shape = (32, 32, 3)

    def load_data(self):
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

        # Scale images to the [0, 1] range
        train_images = train_images.astype("float32") / 255
        test_images = test_images.astype("float32") / 255

        # Make sure images have shape (28, 28, 1)
        self.train_images = np.expand_dims(train_images, -1)
        self.test_images = np.expand_dims(test_images, -1)

        self.train_labels = keras.utils.to_categorical(train_labels, self.num_classes)
        self.test_labels = keras.utils.to_categorical(test_labels, self.num_classes)

        print("train_images shape:", self.train_images.shape)
        print(self.train_images.shape[0], "train samples")
        print(self.test_images.shape[0], "test samples")