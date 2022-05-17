import torch
import torch.nn as nn
from torchvision import datasets, models,transforms as T

class Regnet_x_8gf(nn.Module):
    def __init__(self):
        super(Regnet_x_8gf, self).__init__()
        self.net = models.regnet_x_8gf()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)

class Regnet_y_8gf(nn.Module):
    def __init__(self):
        super(Regnet_y_8gf, self).__init__()
        self.net = models.regnet_y_8gf()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)



class Regnet_x_16gf(nn.Module):
    def __init__(self):
        super(Regnet_x_16gf, self).__init__()
        self.net = models.regnet_x_16gf()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)


class Regnet_y_16gf(nn.Module):
    def __init__(self):
        super(Regnet_y_16gf, self).__init__()
        self.net = models.regnet_y_16gf()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)
