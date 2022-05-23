from datasets.pytorch.dataloaders.rice_health_data_loader import RiceHealthDataLoader
from datasets.pytorch.dataloaders.cotton_disease_data_loader import CottonDiseaseDataLoader
from datasets.pytorch.dataloaders.plant_seedlings import PlantSeedlingDataLoader

data_loaders = {
    "RICE_HEALTH": RiceHealthDataLoader,
    "COTTON_DISEASE": CottonDiseaseDataLoader,
    "PLANT_SEEDLINGS": PlantSeedlingDataLoader
}