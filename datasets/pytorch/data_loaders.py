from datasets.pytorch.dataloaders.rice_health_data_loader import RiceHealthDataLoader
from datasets.pytorch.dataloaders.cotton_disease_data_loader import CottonDiseaseDataLoader
from datasets.pytorch.dataloaders.plant_seedlings import PlantSeedlingDataLoader
from datasets.pytorch.dataloaders.leaf_disease_classification_data_loader import LeafDiseaseDataLoader
from datasets.pytorch.dataloaders.cotton_plants_dataset import CottonPlantsDataLoader

data_loaders = {
    "RICE_HEALTH": RiceHealthDataLoader,
    "COTTON_DISEASE": CottonDiseaseDataLoader,
    "PLANT_SEEDLINGS": PlantSeedlingDataLoader,
    "LEAF_DISEASE": LeafDiseaseDataLoader,
    "COTTON_PLANTS": CottonPlantsDataLoader
}