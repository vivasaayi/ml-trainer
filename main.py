from trainer.pytorch_trainer import PyTorchMLTrainer, models_dictionary as torch_models_disctionary
from datasets.keras.cifar10 import Cifar10DataSet
from torchvision import datasets
from torchvision.transforms import ToTensor

from datasets.pytorch.dataloaders.rice_health_data_loader import RiceHealthDataLoader

from datasets.pytorch.pre_packaged_datasets import datasets as prebuilt_datasets
from datasets.pytorch.data_loaders import data_loaders as supported_data_loaders

ml_trainer = PyTorchMLTrainer("local.cnn")

data_loader_name = "COTTON_DISEASE"
prebuilt_dataset_name = None
dataset_path = "/users/rajanp/Downloads/cotton-disease"
batch_size=64


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