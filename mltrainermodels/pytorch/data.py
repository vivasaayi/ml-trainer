from torchvision import datasets, models,transforms as T
from torch.utils.data import DataLoader
import torch

base_path = "/home/ec2-user/SageMaker/data/rice-diseases-image-dataset/RiceDiseaseDataset"
num_workers=4

transform = T.Compose(
    [
        T.Resize(64),
        T.CenterCrop(64),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def get_training_data_loder(batch_size):
    training_data = datasets.ImageFolder(base_path + '/train', transform=transform)
    # training_data = datasets.MNIST(
    #     root="data",
    #     train=True,
    #     download=True,
    #     transform=transform
    # )
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print(train_dataloader)
    # ds
    return train_dataloader

def get_test_data_loader(batch_size):
    training_data = datasets.ImageFolder(base_path + '/validation', transform=transform)
    # test_data = datasets.MNIST(
    #     root="data",
    #     train=False,
    #     download=True,
    #     transform=transform
    # )
    test_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return test_dataloader
