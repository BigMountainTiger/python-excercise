import os

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, 'xu.jpeg')

img = image.load_img(path, target_size=(224, 224))

plt.imshow(img)
plt.show(block = True)
