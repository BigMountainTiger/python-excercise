from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

tf.logging.set_verbosity(tf.logging.WARN)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
  
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'])
  
    return model

def clone_model(model):
    layers = model.layers
    layers.remove(layers[1])
    
    model = tf.keras.models.Sequential(layers)
  
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'])

    return model

model = create_model()
model.summary()

model.fit(train_images, train_labels, epochs = 5, 
          validation_data = (test_images, test_labels), verbose = 0)

model.save('my_model.h5')


model = keras.models.load_model('my_model.h5')
model.summary()

loss, acc = model.evaluate(test_images, test_labels, verbose = 0)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


model = clone_model(model)
model.summary()
loss, acc = model.evaluate(test_images, test_labels, verbose = 0)
print("Cloned model, accuracy: {:5.2f}%".format(100*acc))


print("\nDONE")



