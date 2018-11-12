from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions as vgg16_decode_predictions

from numpy import array
import matplotlib.pyplot as plt

model = VGG16()

layers = model.layers;
olayer = layers[len(layers) - 1]

weights = olayer.get_weights()
weights = array(weights)
print(len(weights[0]))
print(len(weights[1]))

#print(model.summary())

raw_image = load_img('jingfu.jpeg', target_size=(224, 224))


image = img_to_array(raw_image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

image = vgg16_preprocess_input(image)

yhat = model.predict(image)

print(yhat)

label = vgg16_decode_predictions(yhat)
print(label)

print("\nPrediction:")
label = label[0][0]
print('%s (%.2f%%)' % (label[1], label[2]*100))

print("\nDONE")

plt.imshow(raw_image)
plt.show()

