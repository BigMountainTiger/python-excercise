#https://blog.devcenter.co/developing-a-license-plate-recognition-system-with-machine-learning-in-python-787833569ccd

from skimage.io import imread
from skimage.filters import threshold_otsu

import matplotlib.pyplot as plt

def load_image():
    image_path = "./images/xu.jpeg"
    o_image = imread(image_path, as_gray=False)
    g_image = imread(image_path, as_gray=True)
    gray_image = g_image * 255
    threshold_value = threshold_otsu(gray_image)
    binary_image = gray_image > threshold_value
    
    return o_image, g_image, gray_image, binary_image

if __name__ == '__main__':
    (o_image, g_image, gray_image, binary_image) = load_image()
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(o_image)
    ax2.imshow(binary_image, cmap="gray")
    
    plt.show()
    





