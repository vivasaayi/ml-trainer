import torch
import torch.nn as nn

class SimpleConvnet(nn.Module):
    def __init__(self):
        super(SimpleConvnet, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(5, 5))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(5184, 10)
        self.linear3 = nn.Linear(2000, 10)

    def initialize(self):
        return

    def forward(self, x):
        relu_result = nn.functional.relu(self.conv2d(x))
        flatten_result = self.flatten(relu_result)
        linear_result1 = self.linear1(flatten_result)
        #         linear_result2 = self.linear2(linear_result1)
        #         linear_result3 = self.linear3(linear_result1)
        result = nn.functional.log_softmax(linear_result1, dim=1)
        return result

