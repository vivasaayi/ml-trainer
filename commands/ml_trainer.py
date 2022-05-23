import click

from trainer.keras_trainer import KerasMLTrainer
from trainer.pytorch_trainer import PyTorchMLTrainer, models_dictionary as torch_models_disctionary
from datasets.keras.cifar10 import Cifar10DataSet
from torchvision import datasets
from torchvision.transforms import ToTensor

from datasets.pytorch.dataloaders.rice_health_data_loader import RiceHealthDataLoader

from datasets.pytorch.pre_packaged_datasets import datasets as prebuilt_datasets
from datasets.pytorch.data_loaders import data_loaders as supported_data_loaders


@click.group()
def mltrainer():
    pass

@click.command()
def list_torch_model_names():
    print("Below PyTorch Models are available")
    for key in torch_models_disctionary:
        print(key)

@click.command()
@click.argument('batch_size', default=32)
@click.argument('epoch', default=50)
@click.argument('learning_rate', default=1e-3)
@click.option('--model-name', default='local.simple_convnet')
@click.option('--data-loader-name', default=None)
@click.option('--dataset-path', default=None)
@click.option('--prebuilt-dataset-name', default=None)
@click.option('--model-save-path', default=None)
def train_torch_net(model_name, batch_size, epoch, learning_rate, data_loader_name, dataset_path, prebuilt_dataset_name, model_save_path):
    ml_trainer = PyTorchMLTrainer(model_name)

    if data_loader_name:
        data_loader = supported_data_loaders[data_loader_name](dataset_path)
        training_data_loader = data_loader.get_training_data_loder(batch_size)
        test_data_loader = data_loader.get_test_data_loader(batch_size)
        ml_trainer.initialize_with_data_loader(training_data_loader, test_data_loader)
    else:
        dataset = prebuilt_datasets[prebuilt_dataset_name]
        ml_trainer.initialize_with_dataset(dataset["training_data"], dataset["test_data"], batch_size)

    ml_trainer.train()
    ml_trainer.save_model()

@click.command()
@click.option('--data-loader-name', default=None)
@click.option('--dataset-path', default=None)
def prepare_dataset(data_loader_name, dataset_path):
    print("Preparing DataSet")

    op_train_path = f"{dataset_path}-processed/train"
    op_test_path = f"{dataset_path}-processed/validation"

    print(op_train_path)
    print(op_test_path)

    data_loader = supported_data_loaders[data_loader_name](dataset_path)
    data_loader.preprocess(op_train_path, op_test_path)


mltrainer.add_command(list_torch_model_names)
mltrainer.add_command(train_torch_net)
mltrainer.add_command(prepare_dataset)