import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

def load():
    print("Loading CIFAR Model")

    model = tf.keras.models.load_model('model_results/cifar10')


    print("Evaluating model")
    # sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    # sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

    img_height = 32
    img_width = 32
    img = tf.keras.utils.load_img(
        "/Users/rajanp/Downloads/test/299153.png", target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(np.argmax(score))
    print(np.max(score))

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
