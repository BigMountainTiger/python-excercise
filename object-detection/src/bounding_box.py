import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

path = "../images/xu-big.jpeg"

image = Image.open(path)
image = image.resize((300, 300))
#image = np.array(image, dtype=np.uint8)

fig, ax = plt.subplots(1)
ax.imshow(image)

size = ((60, 100), 150, 150)
rect = patches.Rectangle(size[0], size[1], size[2], linewidth = 1,edgecolor = 'r', facecolor = 'none')
ax.add_patch(rect)

plt.show()