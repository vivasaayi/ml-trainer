import click

from trainer.keras_trainer import KerasMLTrainer
from trainer.pytorch_trainer import PyTorchMLTrainer
from datasets.keras.cotton_disease import CottonDiseaseDataSet
from datasets.pytorch.dataloaders.cotton_disease_data_loader import CottonDiseaseDataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


@click.group()
def cotton():
    pass

@click.group()
def preprocess():
    print('Preprocessing COTTON Dataset')

    pass


@click.group()
def train():
    print('Training COTTON Dataset')

    pass

@click.command()
def prepare_dataset():
    print("Preparing DataSet")

    base_path = "/users/rajanp/Downloads/cotton-disease"
    op_train_path = "/users/rajanp/Downloads/cotton-disease-processed/train"
    op_test_path = "/users/rajanp/Downloads/cotton-disease-processed/validation"

    dataloader = CottonDiseaseDataLoader(base_path)
    dataloader.preprocess(op_train_path, op_test_path)


@click.command()
def keras_cnn():
    cotton_disease_dataset = CottonDiseaseDataSet()
    ml_trainer = KerasMLTrainer("local.cnn", cotton_disease_dataset)
    ml_trainer.train_using_dir(128, 3)
    ml_trainer.save_model()

@click.command()
def keras_simple_convnet():
    cotton_disease_dataset = CottonDiseaseDataSet()
    ml_trainer = KerasMLTrainer("local.simple_convnet", cotton_disease_dataset)
    ml_trainer.train_using_dir(128, 3)
    ml_trainer.save_model()


@click.command()
def torch_simple_convnet():
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
train.add_command(keras_cnn)

train.add_command(torch_simple_convnet)

cotton.add_command(train)

preprocess.add_command(prepare_dataset)
cotton.add_command(preprocess)
