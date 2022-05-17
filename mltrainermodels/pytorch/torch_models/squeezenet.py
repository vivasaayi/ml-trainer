import torch
import torch.nn as nn
from torchvision import datasets, models,transforms as T

class Squeezenet10(nn.Module):
    def __init__(self):
        super(Squeezenet10, self).__init__()
        self.net = models.squeezenet1_0()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)

class Squeezenet11(nn.Module):
    def __init__(self):
        super(Squeezenet11, self).__init__()
        self.net = models.squeezenet1_1()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)
