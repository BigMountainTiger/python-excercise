import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications import resnet50

import matplotlib.pyplot as plt

from song.utilities import imageloader

print("Initiating")

def custom_model():
    base_model = resnet50.ResNet50(input_shape = (224, 224, 3),
        include_top = False, weights = 'imagenet', pooling = 'avg')
    
    x = base_model.output

    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)

    out = keras.layers.Dense(7, activation='softmax', name='output_layer')(x)
    custom_resnet_model = keras.models.Model(inputs=base_model.input,outputs= out)

    for layer in base_model.layers:
        layer.trainable = False


    custom_resnet_model.compile(optimizer=tf.keras.optimizers.Adam(), 
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'])

    return custom_resnet_model


model = custom_model()
model.summary()

from tensorflow.keras.preprocessing import image

def r2a_path(path):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, path)

    return path

def load_raw_image(path):
    path = r2a_path(path)

    img = image.load_img(path, target_size=(224, 224))
    
    return img
    
def load_training_image(path):
    img = load_raw_image(path)
    img = image.img_to_array(img)
    
    return img

def load_training_data():
    train_images = []
    train_labels = [0, 0, 0, 0, 0, 1, 2, 3, 3, 4, 4, 5, 6, 6]
    class_names = ["许勇", "陈景富", "李教授", "老袁", "机器人", "菠萝蜜", "陈庆波"]
        
    train_images.append(load_training_image("images/xu.jpeg"))
    train_images.append(load_training_image("images/xu-1.jpeg"))
    train_images.append(load_training_image("images/xu-2.jpeg"))
    train_images.append(load_training_image("images/xu-3.jpeg"))
    train_images.append(load_training_image("images/xu-4.jpeg"))
    
    train_images.append(load_training_image("images/chen.jpg"))
    train_images.append(load_training_image("images/li.jpeg"))
    
    train_images.append(load_training_image("images/yuan.jpeg"))
    train_images.append(load_training_image("images/yuan-1.jpeg"))
    
    train_images.append(load_training_image("images/toy.jpeg"))
    train_images.append(load_training_image("images/toy-1.jpeg"))
    
    train_images.append(load_training_image("images/jackfruit.jpg"))
    
    train_images.append(load_training_image("images/cheng-qingbo.jpg"))
    train_images.append(load_training_image("images/cheng-qingbo-1.jpg"))
    
    return np.array(train_images), np.array(train_labels), np.array(class_names)

(train_images, train_labels, class_names) = load_training_data()


model.fit(train_images, train_labels, epochs = 5, verbose = 1)

def evaluate(images, labels):
    def getlabel(predictions):
        v = 0
        index = 0;
        
        predictions = predictions[0]
        for i in range(len(predictions)):
            p = predictions[i]
            if (p > v):
                v = p
                index = i
                
        return index
    
    
    for i in range(len(images)):
        image = images[i]
        img = np.expand_dims(image, axis=0)
        predictions = model.predict(img)
        
        label = labels[i]
        if (label != getlabel(predictions)):
            print("Wrong: " + str(label))


#evaluate(train_images, train_labels)

test_loss, test_acc = model.evaluate(train_images, train_labels)

print("test_loss - " + str(test_loss))
print("test_acc - " + str(test_acc))

print("Start predicting")
test_image = r2a_path("images/toy.jpeg")
predictions = model.predict(imageloader.load_training_image(test_image))

def interprate(yhat):
    prediction = yhat[0]
    
    for i in range(len(prediction)):
        p = prediction[i]
        print('%s (%.2f%%)' % (class_names[i], p*100))
        
print(interprate(predictions))


plt.imshow(imageloader.load_raw_image(test_image))
plt.show(block = True)

