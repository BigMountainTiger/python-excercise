import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Total training data size = 60000, test data size = 10000

train_len = 600
test_len = 100

train_images = train_images[0: train_len] / 255.0
train_labels = train_labels[0: train_len]

test_images = test_images[0: test_len] / 255.0
test_labels = test_labels[0: test_len]

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)


def get_model():
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(2, 2),
        activation='relu', input_shape=(28, 28, 1))) 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'])
    
    return model

model = get_model()
model.summary()

model.fit(train_images, train_labels, epochs = 10, verbose = 1)
test_loss, test_acc = model.evaluate(test_images, test_labels)

model.save('../models/fashion_convolution.h5')

print("test_loss - " + str(test_loss))
print("test_acc - " + str(test_acc))


print("\nDone!")





