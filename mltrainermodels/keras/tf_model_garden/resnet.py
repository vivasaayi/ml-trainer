from tensorflow.keras import layers
#import tensorflow_models as tfm

class Resnet18():
    def initialize(self, params):
        print(params)
        self.model = tfm.vision.backbones.ResNet(
            18,
            input_specs = layers.InputSpec(shape=[None, 28, 28, 1]),
        )