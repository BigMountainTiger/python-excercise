import matplotlib.pyplot as plt

import song.modules.image_loader as image_loader


path = "../images/xu.jpeg"

img = image_loader.load_raw_image(path)

plt.figure()
plt.imshow(img)
plt.show()