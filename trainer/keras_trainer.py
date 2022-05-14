from datasets.keras.base_keras_dataset import BaseKerasDataSet

from mltrainermodels.keras.simple_convnet import SimpleConvnet
from mltrainermodels.keras.cnn import CNN

models_dictionary = {
    "local.simple_convnet": SimpleConvnet,
    "local.cnn": CNN
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
        self.x_train = self.dataset.x_train
        self.y_train = self.dataset.y_train
        self.x_test = self.dataset.x_test
        self.y_test = self.dataset.y_test

    def train(self, batch_size:int, epochs:int):
        print(f"Training Keras ML Model {self.model_name}")
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def save_model(self):
        path = f"model_results/dataset_name_tbd/{self.model_name}"
        print("Saving Model to {}".format(path))
        self.model.save(path)