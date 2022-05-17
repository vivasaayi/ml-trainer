import torch
import torch.nn as nn
from torchvision import datasets, models,transforms as T

class Mnasnet0_5(nn.Module):
    def __init__(self):
        super(Mnasnet0_5, self).__init__()
        self.net = models.mnasnet0_5()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)

class Mnasnet1_0(nn.Module):
    def __init__(self):
        super(Mnasnet1_0, self).__init__()
        self.net = models.mnasnet1_0()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)

class Mnasnet1_3(nn.Module):
    def __init__(self):
        super(Mnasnet1_3, self).__init__()
        self.net = models.mnasnet1_3()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)

class Mnasnet0_75(nn.Module):
    def __init__(self):
        super(Mnasnet0_75, self).__init__()
        self.net = models.mnasnet0_75()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)
