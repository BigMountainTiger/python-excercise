import numpy as np
from tensorflow.keras.preprocessing import image

def load_raw_image(path):
    img = image.load_img(path, target_size=(224, 224))
    
    return img
    
def load_training_image(path):
    img = load_raw_image(path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    return img
