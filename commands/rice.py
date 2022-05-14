import click

from trainer.keras_trainer import KerasMLTrainer
from datasets.keras.mnist import MnistDataSet


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

train.add_command(keras_simple_convnet)

rice.add_command(train)