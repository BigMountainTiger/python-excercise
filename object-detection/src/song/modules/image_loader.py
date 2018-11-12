from tensorflow.keras.preprocessing import image

def load_raw_image(path):
    img = image.load_img(path, target_size=(160, 160))
    
    return img

