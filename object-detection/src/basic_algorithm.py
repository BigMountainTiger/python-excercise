import random
import numpy as np

from tensorflow import keras

model_path = "../models/xu.h5"
#model = keras.models.load_model(model_path)

from tensorflow.keras.preprocessing import image

WINDOW_SIZES = [i for i in range(20, 160, 20)]
print(WINDOW_SIZES)


def get_best_bounding_box(img, predict_fn, step=10, window_sizes=WINDOW_SIZES):
    best_box = None
    best_box_prob = -np.inf

    n_predictions = 0;
    # loop window sizes: 20x20, 30x30, 40x40...160x160
    for win_size in window_sizes:
        for top in range(0, img.shape[0] - win_size + 1, step):
            for left in range(0, img.shape[1] - win_size + 1, step):
                box = (top, left, top + win_size, left + win_size)

                cropped_img = img[box[0]:box[2], box[1]:box[3]]

                #print('predicting for box %r' % (box, ))
                n_predictions += 1
                box_prob = predict_fn(cropped_img)
                if box_prob > best_box_prob:
                    best_box = box
                    best_box_prob = box_prob

    print("Total number of predictions - " + str(n_predictions))
    return best_box


def predict_function(x):
    random.seed(x[0][0])
    return random.random()

# 256 x 256
def load_image(path):
    img = image.load_img(path, target_size=(224, 224))
    #img = image.img_to_array(img)
    
    return img

test_image_path = "../images/xu-big.jpeg"
#img = load_image(test_image_path).reshape((224, 224))

img = np.arange(256 * 256).reshape((256, 256))
print(img.shape)

best_box = get_best_bounding_box(img, predict_function)

print('best bounding box %r' % (best_box, ))