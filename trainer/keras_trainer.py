from tensorflow import keras
from datasets.keras.base_keras_dataset import BaseKerasDataSet

from mltrainermodels.keras.simple_convnet import SimpleConvnet
from mltrainermodels.keras.cnn import CNN

from mltrainermodels.keras.tf_model_garden.resnet import Resnet18

from mltrainermodels.keras.keras_applications.resnet import Resnet50

models_dictionary = {
    "local.simple_convnet": SimpleConvnet,
    "local.cnn": CNN,

    # Models from Tensorflow Models library
    "tfm.resnet18": Resnet18,

    # Models from Keras Applications
    "keras_applications.resnet50": Resnet50
}

class KerasMLTrainer():
    def __init__(self, model_name:str, dataset: BaseKerasDataSet):
        self.model_name = model_name
        print(f"Initializing Keras ML Trainer {self.model_name}")

        self.dataset = dataset
        self.model_provider = models_dictionary[model_name]()
        self.model_provider.initialize({
            "input_shape": dataset.input_shape,
            "num_classes": dataset.num_classes
        })
        self.model = self.model_provider.model

        print(f"Model Initialized {self.model_name}")
        self.model.summary()

    def initialize(self):
        self.dataset.load_data()
        self.train_images = self.dataset.train_images
        self.train_labels = self.dataset.train_labels
        self.test_images = self.dataset.test_images
        self.test_labels = self.dataset.test_labels

    def train_using_dir(self, batch_size:int, epochs:int):
        self.dataset.load_data()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x=self.dataset.train_data, validation_data=self.dataset.test_data, epochs=50)

    def train(self, batch_size:int, epochs:int):
        print(f"Training Keras ML Model {self.model_name}")
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.fit(self.train_images, self.train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    def evaluate(self):
        score = self.model.evaluate(self.test_images, self.test_labels, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def save_model(self):
        path = f"model_results/dataset_name_tbd/{self.model_name}"
        print("Saving Model to {}".format(path))
        self.model.save(path)