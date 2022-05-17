from datasets.pytorch.dataloaders.rice_health_data_loader import RiceHealthDataLoader
from datasets.pytorch.dataloaders.cotton_disease_data_loader import CottonDiseaseDataLoader

data_loaders = {
    "RICE_HEALTH": RiceHealthDataLoader,
    "COTTON_DISEASE": CottonDiseaseDataLoader
}