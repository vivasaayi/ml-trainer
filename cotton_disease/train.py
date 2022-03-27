from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

train_datagenerator = ImageDataGenerator(rescale = 1.0/255,
                                        shear_range = 0.2,
                                        zoom_range = 0.5,
                                        horizontal_flip = True,
                                        rotation_range=10,
                                        width_shift_range=0.2,
                                        brightness_range=[0.2,1.2]
                                        )
test_datagenerator = ImageDataGenerator(rescale = 1.0/255)

def train():
    print("Training Cotton Disase Model")
    train_data = train_datagenerator.flow_from_directory('cotton_disease/dataset/train',
                                                         target_size=(256, 256),
                                                         batch_size=32,
                                                         class_mode='categorical')

    test_data = test_datagenerator.flow_from_directory('cotton_disease/dataset/val',
                                                       target_size=(227, 227),
                                                       batch_size=64,
                                                       class_mode='categorical')

    cnn = tf.keras.models.Sequential()
    # Convolution
    cnn.add(
        tf.keras.layers.Conv2D(filters=64, padding="same", kernel_size=3, activation='relu', input_shape=[227, 227, 3]))
    cnn.add(tf.keras.layers.Conv2D(filters=32, padding="same", kernel_size=3, activation='relu'))
    # pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    # Convolution
    cnn.add(tf.keras.layers.Conv2D(filters=16, padding="same", kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.Conv2D(filters=16, padding="same", kernel_size=3, activation='relu'))
    # pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # flaterning
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    # Output layer
    cnn.add(tf.keras.layers.Dense(units=4, activation='softmax'))

    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = cnn.fit(x=train_data, validation_data=test_data, epochs=50)