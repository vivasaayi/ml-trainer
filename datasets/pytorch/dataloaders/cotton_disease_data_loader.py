from torchvision import datasets, models,transforms as T
from torch.utils.data import DataLoader

class CottonDiseaseDataLoader():
    def __init__(self, base_path):
        self.base_path = base_path

    def get_transforms(self):
        transforms = T.Compose(
            [
                T.Resize(128),
                T.CenterCrop(128),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        return transforms

    def get_training_data_loder(self, batch_size):
        training_data = datasets.ImageFolder(self.base_path + '/train', transform=self.get_transforms())
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        return train_dataloader

    def get_test_data_loader(self, batch_size):
        test_data = datasets.ImageFolder(self.base_path + '/validation', transform=self.get_transforms())
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        return test_dataloader
