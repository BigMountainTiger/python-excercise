import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Total training data size = 60000, test data size = 10000

train_len = 2000
test_len = 10

train_images = train_images[0: train_len] / 255.0
train_labels = train_labels[0: train_len]

test_images = test_images[0: test_len] / 255.0
test_labels = test_labels[0: test_len]

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

image = np.array([train_images[0] / 255.0])
image = image.reshape(1, 28, 28, 1)

model_path = "../models/fashion_convolution.h5"
model = keras.models.load_model(model_path)

predictions = model.predict(image)

def interprate(yhat):
    prediction = yhat[0]
    
    for i in range(len(prediction)):
        p = prediction[i]
        print('%s (%.2f%%)' % (class_names[i], p*100))
        
print(interprate(predictions))

plt.figure()
plt.imshow(image[0].reshape(28, 28))
plt.colorbar()
plt.grid(False)
plt.show()

print("\nDone!")





