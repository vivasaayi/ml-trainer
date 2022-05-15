import click

from trainer.keras_trainer import KerasMLTrainer
from trainer.pytorch_trainer import PyTorchMLTrainer
from datasets.keras.cifar10 import Cifar10DataSet
from torchvision import datasets
from torchvision.transforms import ToTensor


@click.group()
def rice():
    pass

@click.group()
def train():
    print('Training RICE Dataset')

    pass

@click.command()
def torch_simple_convnet():
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    ml_trainer = PyTorchMLTrainer("local.simple_convnet", training_data, test_data)
    ml_trainer.initialize()
    ml_trainer.train()
    ml_trainer.save_model()


@click.command()
def torch_resnet18():
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    ml_trainer = PyTorchMLTrainer("torch_models.resnet18", training_data, test_data)
    ml_trainer.initialize()
    ml_trainer.train()
    ml_trainer.save_model()


train.add_command(keras_cnn)
train.add_command(keras_simple_convnet)

train.add_command(keras_ka_resnet50)

train.add_command(torch_simple_convnet)
train.add_command(torch_resnet18)

rice.add_command(train)