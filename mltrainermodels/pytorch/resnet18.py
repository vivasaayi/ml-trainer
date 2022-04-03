import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, models,transforms as T
import matplotlib.pyplot as plt
from torchinfo import summary
import numpy as np
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader

net =  models.resnet18()

summary(net, input_size=(1,3,28,28))

if torch.cuda.is_available():
    net.cuda()

transform = T.Compose([T.Grayscale(num_output_channels=3), T.ToTensor()])

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


learning_rate = 1e-3
batch_size = 64
epochs = 100

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        # X = X.cuda()
        # y = y.cuda()
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            # X = X.cuda()
            # y = y.cuda()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


train_loop(train_dataloader,net,loss_fn,optimizer)
test_loop(test_dataloader,net,loss_fn)