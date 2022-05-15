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


