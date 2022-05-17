import torch
import torch.nn as nn
from torchvision import datasets, models,transforms as T

class Shufflenet_v2_x0_5(nn.Module):
    def __init__(self):
        super(Shufflenet_v2_x0_5, self).__init__()
        self.net = models.shufflenet_v2_x0_5()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)

class Shufflenet_v2_x1_0(nn.Module):
    def __init__(self):
        super(Shufflenet_v2_x1_0, self).__init__()
        self.net = models.shufflenet_v2_x1_0()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)



class Shufflenet_v2_x1_5(nn.Module):
    def __init__(self):
        super(Shufflenet_v2_x1_5, self).__init__()
        self.net = models.shufflenet_v2_x1_5()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)


class Shufflenet_v2_x2_0(nn.Module):
    def __init__(self):
        super(Shufflenet_v2_x2_0, self).__init__()
        self.net = models.shufflenet_v2_x2_0()

        if torch.cuda.is_available():
            self.net.cuda()

    def initialize(self):
        return

    def forward(self, x):
        return self.net.forward(x)
