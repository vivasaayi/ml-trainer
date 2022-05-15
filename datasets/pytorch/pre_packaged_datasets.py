from torchvision import datasets
from torchvision.transforms import ToTensor

datasets = {
    "MNIST": {
        "training_data": datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        ),
        "test_data": datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )
    }
}