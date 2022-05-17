import torch
import torch.nn as nn
from torchvision import datasets, models,transforms as T

class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.net = models.inception_v3()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)
