import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import rgb2gray

image_path = "./images/car-3.jpeg"

image = imread(image_path)
g_image = rgb2gray(image)

thresh = threshold_otsu(g_image)
bw = closing(g_image > thresh, square(3))

cleared = clear_border(bw)
label_image = label(cleared)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image)

regions = regionprops(label_image)
for region in regions:

    if region.area >= 1:

        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr),
            maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()

plt.show()




