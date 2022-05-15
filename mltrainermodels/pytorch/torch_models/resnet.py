import torch
import torch.nn as nn
from torchvision import datasets, models,transforms as T

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.net = models.resnet18()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)


class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        self.net = models.resnet34()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)

class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.net = models.resnet50()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)

class Resnet101(nn.Module):
    def __init__(self):
        super(Resnet101, self).__init__()
        self.net = models.resnet101()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)

class Resnet152(nn.Module):
    def __init__(self):
        super(Resnet152, self).__init__()
        self.net = models.resnet152()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)

