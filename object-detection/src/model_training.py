import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications import resnet50

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

print("Initiating")

def custom_model():
    base_model = resnet50.ResNet50(input_shape = (224, 224, 3),
        include_top = False, weights = 'imagenet', pooling = 'avg')
    
    x = base_model.output

    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)

    out = keras.layers.Dense(2, activation='softmax', name='output_layer')(x)
    custom_resnet_model = keras.models.Model(inputs=base_model.input,outputs= out)

    for layer in base_model.layers:
        layer.trainable = False


    custom_resnet_model.compile(optimizer=tf.keras.optimizers.Adam(), 
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'])

    return custom_resnet_model


model = custom_model()

def load_raw_image(path):
    img = image.load_img(path, target_size=(224, 224))
    
    return img
    
def load_training_image(path):
    img = load_raw_image(path)
    img = image.img_to_array(img)
    
    return img

def load_test_image(path):
    img = np.expand_dims(load_training_image(path), axis=0)
    
    return img

def load_training_data():
    train_images = []
    train_labels = [0, 1, 1, 1]
    class_names = ["许勇", "Other"]
        
    train_images.append(load_training_image("../images/xu.jpeg"))
    
    train_images.append(load_training_image("../images/other-1.jpeg"))
    train_images.append(load_training_image("../images/other-2.jpeg"))
    train_images.append(load_training_image("../images/other-3.jpeg"))

    
    return np.array(train_images), np.array(train_labels), np.array(class_names)

(train_images, train_labels, class_names) = load_training_data()


model.fit(train_images, train_labels, epochs = 5, verbose = 1)

model_path = "../models/xu.h5"
model.save(model_path)

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
test_image = "../images/xu.jpeg"
model = keras.models.load_model(model_path)
predictions = model.predict(load_test_image(test_image))

def interprate(yhat):
    prediction = yhat[0]
    
    for i in range(len(prediction)):
        p = prediction[i]
        print('%s (%.2f%%)' % (class_names[i], p*100))
        
print(interprate(predictions))


plt.imshow(load_raw_image(test_image))
plt.show(block = True)

