import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


a_len = 60
t_len = 10

train_images = train_images[0: a_len]
train_labels = train_labels[0: a_len]

test_images = test_images[0: t_len]
test_labels = test_labels[0: t_len]

print(train_images.shape)
print(len(train_labels))
print(train_labels)

print("\n")
print(test_images.shape)
print(len(test_labels))

'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''

train_images = train_images / 255.0
test_images = test_images / 255.0

'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs = 5, verbose = 0)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("test_loss - " + str(test_loss))
print("test_acc - " + str(test_acc))

predictions = model.predict(test_images[0: 1])

print(str(np.argmax(predictions[0])) + " - " + str(test_labels[0]))

print("\ndone")



