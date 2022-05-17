import torch
import torch.nn as nn

# 128 ->
# 200 ->
# 400 ->
# 600 ->

class SimpleConvnet(nn.Module):
    def __init__(self, params={}):
        super(SimpleConvnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=(5, 5))
        self.max_pool_2d = nn.MaxPool2d((5, 5))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(.5)

        num_classes = 4
        if(num_classes in params):
            num_classes = params["num_classes"]

        self.linear1 = nn.Linear(288, num_classes)

    def initialize(self):
        return

    def forward(self, x):
        # Convolution 1
        conv1_result = self.conv1(x)
        relu1_result = nn.functional.relu(conv1_result)

        # Maxpool 2d
        max_pool_2d_result = self.max_pool_2d(relu1_result)

        # Convolution 2
        conv2_result = self.conv2(max_pool_2d_result)
        relu2_result = nn.functional.relu(conv2_result)

        # Maxpool 2d
        max_pool_2d_result = self.max_pool_2d(relu2_result)

        # Flatten
        flatten_result = self.flatten(max_pool_2d_result)

        # Linear
        linear_result1 = self.linear1(flatten_result)

        # Dropout
        dropout_result = self.dropout(linear_result1)

        # Softmax
        result = nn.functional.log_softmax(dropout_result, dim=1)
        return result

