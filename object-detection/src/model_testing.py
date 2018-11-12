import numpy as np

from tensorflow import keras

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


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

model_path = "../models/xu.h5"

print("Loading model")
test_image_path = "../images/xu.jpeg"
model = keras.models.load_model(model_path)

print("Start predicting")
test_image_path = "../images/xu-big.jpeg"
img = image.load_img(test_image_path, target_size=(300, 300))

def crop_image(img, size):
    img = img.crop(size)
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    return img

test_img = crop_image(img, (0, 0, 50, 50))
predictions = model.predict(test_img)

print(predictions[0][0])

class_names = ["许勇", "Other"]
def interprate(yhat):
    prediction = yhat[0]
    
    for i in range(len(prediction)):
        p = prediction[i]
        print('%s (%.2f%%)' % (class_names[i], p*100))
        
print(interprate(predictions))


plt.imshow(img)
plt.show(block = True)

