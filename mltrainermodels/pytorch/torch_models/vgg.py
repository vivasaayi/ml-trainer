import torch
import torch.nn as nn
from torchvision import datasets, models,transforms as T

class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.net = models.vgg11()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)


class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        self.net = models.vgg13()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.net = models.vgg16()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.net = models.vgg19()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)
