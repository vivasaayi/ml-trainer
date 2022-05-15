from tensorflow import keras

class Resnet50():
    def initialize(self, params):
        print(params)
        self.model = keras.applications.ResNet50(weights=None,input_shape=(32,32,3))