import click

from trainer.keras_trainer import KerasMLTrainer
from trainer.pytorch_trainer import PyTorchMLTrainer
from datasets.keras.mnist import MnistDataSet
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
def keras_simple_convnet():
    mnist_dataset = MnistDataSet()
    ml_trainer = KerasMLTrainer("local.simple_convnet", mnist_dataset)
    ml_trainer.initialize()
    ml_trainer.train(128, 3)
    ml_trainer.save_model()

@click.command()
def torch_simple_convnet():
    #mnist_dataset = PyTorchMLTrainer()
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    ml_trainer = PyTorchMLTrainer("local.simple_convnet", training_data, test_data)
    ml_trainer.initialize()
    ml_trainer.train()
    ml_trainer.save_model()


train.add_command(keras_simple_convnet)
train.add_command(torch_simple_convnet)

rice.add_command(train)