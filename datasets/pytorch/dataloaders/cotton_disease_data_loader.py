import os
from torchvision import datasets, models,transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import save_image

class CottonDiseaseDataLoader():
    def __init__(self, base_path):
        self.base_path = base_path

    def get_transforms(self):
        transforms = T.Compose(
            [
                T.Resize(299),
                T.CenterCrop(299),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        return transforms

    def get_empty_transform(self):
        transforms = T.Compose(
            [
                #T.Resize(299),
                #T.CenterCrop(299),
                T.ToTensor(),
                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        return transforms

    def get_training_data_loder(self, batch_size, transform=None):
        if transform is None:
            transform = self.get_empty_transform()
        training_data = datasets.ImageFolder(self.base_path + '/train', transform=transform)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        return train_dataloader

    def get_test_data_loader(self, batch_size, transform=None):
        test_data = datasets.ImageFolder(self.base_path + '/validation', transform=transform)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        return test_dataloader

    def preprocess(self, transformed_train_path, transformed_test_path):
        tx = self.get_transforms()

        training_data_loader = self.get_training_data_loder(32, transform=tx)
        test_data_loader = self.get_test_data_loader(32, transform=tx)

        classes = training_data_loader.dataset.classes

        file_index = 0
        for batch, (X, y) in enumerate(training_data_loader):
            index = 0
            print(len(X), y)
            for t in X:
                path = os.path.join(transformed_train_path, classes[y[index]])
                os.makedirs(path, exist_ok=True)
                # print(path)
                save_image(t, os.path.join(path, f"{file_index}.jpg"))
                index = index + 1
                file_index = file_index + 1

            print(f"FILE INDEX>>>{file_index}")

        file_index = 0
        for batch, (X, y) in enumerate(test_data_loader):
            index = 0
            print(len(X), y)
            for t in X:
                path = os.path.join(transformed_test_path, classes[y[index]])
                os.makedirs(path, exist_ok=True)
                # print(path)
                save_image(t, os.path.join(path, f"{file_index}.jpg"))
                index = index + 1
                file_index = file_index + 1

            print(f"FILE INDEX>>>{file_index}")