import sys

import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

import matplotlib.pyplot as plt

model = VGG16()

def clone_model(model):
    layers = model.layers
    layers.pop()
    
    for l in layers:
        l.trainable = False
        
    olayer = keras.layers.Dense(3, activation=tf.nn.softmax, name="Prediction")
    layers.append(olayer)
    
    model = tf.keras.models.Sequential(layers)
  
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'])

    return model

def get_training_data():
    train_images = []
    train_labels = [0, 1, 2]
    class_names = ["许勇", "机器人", "Jingfu"]
    
    def get_image(name):
        image = load_img(name, target_size=(224, 224))
        image = img_to_array(image)
        
        return image
        
    train_images.append(get_image("xu.jpeg"))
    train_images.append(get_image("xu-toy.jpeg"))
    train_images.append(get_image("jingfu.jpeg"))
    
    return np.array(train_images), np.array(train_labels), np.array(class_names)
    
(train_images, train_labels, class_names) = get_training_data()

model = clone_model(model)
model.fit(train_images, train_labels, epochs = 5, verbose = 1)


def get_test_image(name):
    raw_image = load_img(name, target_size=(224, 224))
    image = img_to_array(raw_image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = vgg16_preprocess_input(image)
    
    return image, raw_image

def interprate(yhat):
    prediction = yhat[0]
    
    for i in range(len(prediction)):
        p = prediction[i]
        print('%s (%.2f%%)' % (class_names[i], p*100))
        

(test_image, raw_image) = get_test_image('jingfu.jpeg')

yhat = model.predict(test_image)

print("\n")
interprate(yhat)


print("\nDONE")

plt.imshow(raw_image)
plt.show(block = True)

