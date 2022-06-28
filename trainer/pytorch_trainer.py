from mltrainermodels.pytorch.simple_convnet import SimpleConvnet
from mltrainermodels.pytorch.cnn import CNN
from mltrainermodels.pytorch.torch_models.resnet import Resnet18, Resnet34, Resnet50, Resnet101, Resnet152
from mltrainermodels.pytorch.torch_models.vgg import VGG11, VGG13, VGG16, VGG19
from mltrainermodels.pytorch.torch_models.googlenet import GoogleNet
from mltrainermodels.pytorch.torch_models.inception import InceptionV3
from mltrainermodels.pytorch.torch_models.resnext import Resnext50, Resnext101
from mltrainermodels.pytorch.torch_models.squeezenet import Squeezenet10, Squeezenet11
from mltrainermodels.pytorch.torch_models.shufflenet import Shufflenet_v2_x1_5, Shufflenet_v2_x0_5, Shufflenet_v2_x1_0, \
    Shufflenet_v2_x2_0
from mltrainermodels.pytorch.torch_models.mnasnet import Mnasnet0_5, Mnasnet1_0, Mnasnet1_3, Mnasnet0_75
from mltrainermodels.pytorch.torch_models.mobilenet import MobileNetV3Large, MobileNetV3Small

from mltrainermodels.pytorch.torch_models.regnet import Regnet_x_8gf, Regnet_x_16gf, Regnet_y_8gf, Regnet_y_16gf

from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


from torchinfo import summary
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

models_dictionary = {
    "local.simple_convnet": SimpleConvnet,
    "local.cnn": CNN,

    "torch_models.resnet18": Resnet18,
    "torch_models.resnet34": Resnet34,
    "torch_models.resnet50": Resnet50,
    "torch_models.resnet101": Resnet101,
    "torch_models.resnet152": Resnet152,

    "torch_models.vgg11": VGG11,
    "torch_models.vgg13": VGG13,
    "torch_models.vgg16": VGG16,
    "torch_models.vgg19": VGG19,

    "torch_models.googlenet": GoogleNet,

    "torch_models.inceptionv3": InceptionV3,

    "torch_models.resnext50": Resnext50,
    "torch_models.resnext101": Resnext101,

    "torch_models.squeezenet10": Squeezenet10,
    "torch_models.squeezenet11": Squeezenet11,

    "torch_models.shufflenet_v2_x1_5": Shufflenet_v2_x1_5,
    "torch_models.shufflenet_v2_x0_5": Shufflenet_v2_x0_5,
    "torch_models.shufflenet_v2_x1_0": Shufflenet_v2_x1_0,
    "torch_models.shufflenet_v2_x2_0": Shufflenet_v2_x2_0,

    "torch_models.mnasnet0_5": Mnasnet0_5,
    "torch_models.mnasnet1_0": Mnasnet1_0,
    "torch_models.mnasnet1_3": Mnasnet1_3,
    "torch_models.mnasnet0_75": Mnasnet0_75,

    "torch_models.mobilenetv3large": MobileNetV3Large,
    "torch_models.mobilenetv3small": MobileNetV3Small,

    "torch_models.regnet_x_8gf": Regnet_x_8gf,
    "torch_models.regnet_x_16gf": Regnet_x_16gf,
    "torch_models.regnet_y_8gf": Regnet_y_8gf,
    "torch_models.regnet_y_16gf": Regnet_y_16gf
}

index = 0;


class PyTorchMLTrainer():
    def __init__(self, model_name):
        self.model_name = model_name
        print(f"Initializing Keras ML Trainer {self.model_name}")

        self.model = models_dictionary[model_name]()
        self.model.initialize()

        if torch.cuda.is_available():
            print("Moving Model to CUDA")
            self.model.cuda()

        print(f"Model Initialized {self.model_name}")

        self.use_logits_for_loss_function = False
        if (self.model_name == "torch_models.googlenet" or self.model_name == "torch_models.inceptionv3"):
            self.use_logits_for_loss_function = True

        # ToDO: FIX ME
        summary(self.model)

    def initialize_with_dataset(self, training_data, test_data, batch_size):
        training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        self.initialize_with_data_loader(training_data_loader, test_dataloader)

    def initialize_with_data_loader(self, training_data_loader, test_data_loader):
        self.learning_rate = 1e-1
        self.batch_size = 32
        self.epochs = 200

        self.train_dataloader = training_data_loader
        self.test_dataloader = test_data_loader

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def appendl_line(self, data):
        print(data)
        file1 = open("result.txt", "a")  # append mode
        file1.write(data)
        file1.close()

    def train_loop(self):
        size = len(self.train_dataloader.dataset)
        for batch, (X, y) in enumerate(self.train_dataloader):
            # Compute prediction and loss
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            pred = self.model(X)

            if self.use_logits_for_loss_function:
                loss = self.loss_fn(pred.logits, y)
            else:
                loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                self.appendl_line(f"\n\nloss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n\n")

    def test_loop(self):
        global index
        size = len(self.test_dataloader.dataset)
        test_loss, correct = 0, 0

        y_pred = []
        y_true = []

        with torch.no_grad():
            for X, y in self.test_dataloader:
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()
                pred = self.model(X)

                # print(pred)
                if self.use_logits_for_loss_function:
                    output = (torch.max(torch.exp(pred.logits), 1)[1]).data.cpu().numpy()
                else:
                    output = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()

                labels = y.data.cpu().numpy()

                # print("Predicted:", output)
                # print("Actual:", labels)

                y_pred.extend(output)
                y_true.extend(labels)

                # print("Actual:", y_true)
                # print("Predicted:", y_pred)

                if self.use_logits_for_loss_function:
                    test_loss += self.loss_fn(pred.logits, y).item()
                    correct += (pred.logits.argmax(1) == y).type(torch.float).sum().item()
                else:
                    test_loss += self.loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size

        self.appendl_line(f"\n\nTest Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n\n")

        classes = self.train_dataloader.dataset.classes

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)

        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 100, index=[i for i in classes],
                             columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig("trainingresults/" + str(index) + '_output.png')
        index = index + 1
        plt.show()
        plt.close()

        self.appendl_line(df_cm.to_string() + "\n\n")



    def train(self):
        print(f"Training ML Model {self.model_name}")
        for epoch in range(0, self.epochs):
            self.appendl_line(f"Starting epoch {epoch}")
            self.train_loop()
            self.test_loop()
            self.appendl_line(f"Completed epoch {epoch}")

    def save_model(self):
        print("Saving Model")
