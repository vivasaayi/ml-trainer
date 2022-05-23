import torch
import torch.nn as nn

# 128 ->
# 200 ->
# 400 ->
# 600 ->

class CNN(nn.Module):
    def __init__(self, params={}):
        num_classes = 38
        if (num_classes in params):
            num_classes = params["num_classes"]

        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(5, 5))
        self.maxpool1 = nn.MaxPool2d((5, 5))


        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(5, 5))
        self.maxpool2 = nn.MaxPool2d((5, 5))

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(4840, 484)

        self.linear2 = nn.Linear(484, num_classes)

    def initialize(self):
        return

    def forward(self, x):
        # Convolution 1
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)


        # Convolution 2
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.maxpool2(x)

        # Flatten
        x = self.flatten(x)

        # Linear 1
        x = self.linear1(x)
        x = nn.functional.relu(x)

        # Linear 2
        x = self.linear2(x)

        # Softmax
        result = nn.functional.log_softmax(x, dim=1)
        return result

