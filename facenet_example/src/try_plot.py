import matplotlib.pyplot as plt
from matplotlib import font_manager
from tensorflow.keras.preprocessing import image


def load_raw_image(path):
    img = image.load_img(path, target_size=(160, 160))
    
    return img

img1 = load_raw_image("./processed-images/xu-1.jpeg")
img2 = load_raw_image("./processed-images/xu-2.jpeg")

title1 = "人工智能认定是同一个罪犯"
title2 = "\n欧几里德距离 - " + str(2.3)

font_path = "./model/simhei.ttf"
prop = font_manager.FontProperties(fname=font_path)

fig = plt.figure()
fig.set_size_inches(6, 4)
fig.suptitle(title1 + title2, fontproperties=prop)
plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.show()
