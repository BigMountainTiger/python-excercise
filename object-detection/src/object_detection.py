import numpy as np

from tensorflow import keras

import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

WINDOW_SIZES = [120]

def crop_image(img, size):
    img = img.crop(size)
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_function(img):
    predictions = model.predict(img)
    
    return predictions[0][0]

def get_best_bounding_box(img, predict_fn, step=10, window_sizes=WINDOW_SIZES):
    best_box = None
    best_box_prob = -np.inf

    # loop window sizes: 20x20, 30x30, 40x40...160x160
    for win_size in window_sizes:
        for top in range(0, 300 - win_size + 1, step):
            for left in range(0, 300 - win_size + 1, step):
                
                box = (top, left, top + win_size, left + win_size)
                cropped_img = crop_image(img, box)

                box_prob = predict_fn(cropped_img)
                if box_prob > best_box_prob:
                    best_box = box
                    best_box_prob = box_prob

    return best_box

best_box = get_best_bounding_box(img, predict_function)

print(best_box)

fig, ax = plt.subplots(1)
ax.imshow(img)

size = ((best_box[0], best_box[1]), best_box[2] - best_box[0], best_box[3] - best_box[1])
rect = patches.Rectangle(size[0], size[1], size[2], linewidth = 1,edgecolor = 'r', facecolor = 'none')
ax.add_patch(rect)

plt.show()
