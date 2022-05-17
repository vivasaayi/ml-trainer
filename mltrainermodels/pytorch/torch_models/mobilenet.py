import torch
import torch.nn as nn
from torchvision import datasets, models,transforms as T

class MobileNetV3Large(nn.Module):
    def __init__(self):
        super(MobileNetV3Large, self).__init__()
        self.net = models.mobilenet_v3_large()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)

class MobileNetV3Small(nn.Module):
    def __init__(self):
        super(MobileNetV3Small, self).__init__()
        self.net = models.mobilenet_v3_small()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)
