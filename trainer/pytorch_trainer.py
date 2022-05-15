from mltrainermodels.pytorch.simple_convnet import SimpleConvnet
from mltrainermodels.pytorch.torch_models.resnet import Resnet18

from torchinfo import summary
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

models_dictionary = {
    "local.simple_convnet": SimpleConvnet,

    "torch_models.resnet18": Resnet18
}

class PyTorchMLTrainer():
    def __init__(self, model_name, training_data, test_data):
        self.model_name = model_name
        print(f"Initializing Keras ML Trainer {self.model_name}")

        self.training_data = training_data
        self.test_data = test_data
        self.model = models_dictionary[model_name]()
        self.model.initialize()

        if torch.cuda.is_available():
            self.model.cuda()

        print(f"Model Initialized {self.model_name}")
        summary(self.model)

    def initialize(self):
        self.learning_rate = 1e-1
        self.batch_size = 64
        self.epochs = 200

        self.train_dataloader = DataLoader(self.training_data, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train_loop(self):
        size = len(self.train_dataloader.dataset)
        for batch, (X, y) in enumerate(self.train_dataloader):
            # Compute prediction and loss
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(self):
        size = len(self.test_dataloader.dataset)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def train(self):
        print(f"Training ML Model {self.model_name}")
        self.train_loop()
        self.test_loop()

    def save_model(self):
        print("Saving Model")