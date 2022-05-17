import torch
import torch.nn as nn
from torchvision import datasets, models,transforms as T

class Resnext50(nn.Module):
    def __init__(self):
        super(Resnext50, self).__init__()
        self.net = models.resnext50_32x4d()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)


class Resnext101(nn.Module):
    def __init__(self):
        super(Resnext101, self).__init__()
        self.net = models.resnext101_32x8d()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)
