import numpy as np
from tensorflow import keras
from datasets.keras.base_keras_dataset import BaseKerasDataSet

from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CottonDiseaseDataSet(BaseKerasDataSet):
    def __init__(self):
        super().__init__()
        print("Initializing COTTONDISEASE Dataset")
        self.num_classes = 4
        self.input_shape = (256, 256, 3)

    def load_data(self):
        train_datagenerator = ImageDataGenerator(rescale=1.0 / 255,
                                                 shear_range=0.2,
                                                 zoom_range=0.5,
                                                 horizontal_flip=True,
                                                 rotation_range=10,
                                                 width_shift_range=0.2,
                                                 brightness_range=[0.2, 1.2]
                                                 )
        test_datagenerator = ImageDataGenerator(rescale=1.0 / 255)

        self.train_data = train_datagenerator.flow_from_directory('data/cotton-disease/train',
                                                             target_size=(256, 256),
                                                             batch_size=32,
                                                             class_mode='categorical')

        self.test_data = test_datagenerator.flow_from_directory('data/cotton-disease/val',
                                                           target_size=(227, 227),
                                                           batch_size=64,
                                                           class_mode='categorical')