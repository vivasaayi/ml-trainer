
from datasets.keras.cotton_disease import CottonDiseaseDataSet
from trainer.keras_trainer import KerasMLTrainer

cotton_disease_dataset = CottonDiseaseDataSet()
ml_trainer = KerasMLTrainer("local.cnn", cotton_disease_dataset)
ml_trainer.train_using_dir(128, 3)
ml_trainer.save_model()