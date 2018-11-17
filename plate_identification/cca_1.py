from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches

import matplotlib.pyplot as plt

from file_loader import load_image

(o_image, g_image, gray_image, binary_image) = load_image()

if __name__ == '__main__':
    label_image = measure.label(binary_image)
    fig, (ax1) = plt.subplots(1)
    ax1.imshow(o_image, cmap="gray");
    
    regions = regionprops(label_image)
    print(len(regions))
    for region in regions:
        area = region.area
        if area < 50:
            print("Smaller than 50")
            continue
    
        # the bounding box coordinates
        minRow, minCol, maxRow, maxCol = region.bbox
        rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow,
                                       edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rectBorder)
    
    plt.show()



