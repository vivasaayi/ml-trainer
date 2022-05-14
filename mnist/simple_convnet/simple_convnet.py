import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# https://keras.io/examples/vision/mnist_convnet/

class SimpleConvnet():
    def __init__(self):
        print("Initializing Simple Convnet")

        # Model / data parameters
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

    def build_model(self):
        print("Building Model")
        self.model = keras.Sequential(
            [
                keras.Input(shape=self.input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        self.model.summary()

    def train(self):
        batch_size = 128
        epochs = 3

        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def save_model(self):
        path = "model_results/mnist.py/simple_convnet"
        print("Saving Model to {}".format(path))
        self.model.save(path)

