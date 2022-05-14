class PyTorchMLTrainer():
    def __init__(self, model_name, training_data, test_data):
        self.model_name = model_name

        print(f"Initializing ML Trainer {self.model_name}")

        self.training_data = training_data
        self.test_data = test_data

    def train(self):
        print(f"Training ML Model {self.model_name}")