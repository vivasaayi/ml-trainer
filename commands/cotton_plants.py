import click

from trainer.keras_trainer import KerasMLTrainer
from trainer.pytorch_trainer import PyTorchMLTrainer
from datasets.keras.cotton_disease import CottonDiseaseDataSet
from datasets.pytorch.dataloaders.cotton_disease_data_loader import CottonDiseaseDataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


@click.group()
def cottonplants():
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

    base_path = "/Users/rajanp/extracted_datasets_local"
    op_train_path = "/Users/rajanp/extracted_datasets_local-processed/train"
    op_test_path = "/Users/rajanp/extracted_datasets_local-processed/validation"

    dataloader = CottonDiseaseDataLoader(base_path)
    dataloader.preprocess(op_train_path, op_test_path)

preprocess.add_command(prepare_dataset)
cottonplants.add_command(preprocess)
