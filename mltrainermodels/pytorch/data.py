from torchvision import datasets, models,transforms as T
from torch.utils.data import DataLoader

base_path = "/Users/rajanp/Downloads/rice/RiceDiseaseDataset"
num_workers=32
img_size=28

transform = T.Compose(
    [
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def get_training_data_loder(batch_size):
    kwargs = {}
    training_data = datasets.ImageFolder(base_path + '/train', transform=transform)
    # training_data = datasets.MNIST(
    #     root="data",
    #     train=True,
    #     download=True,
    #     transform=transform
    # )
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, **kwargs)
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
    test_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    return test_dataloader
