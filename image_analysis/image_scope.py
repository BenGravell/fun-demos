import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image

IMAGE_DIR_PATH = Path(__file__).parent / "images_tiny"


# Define the image names and paths
image_names = ['mona_lisa', 'starry_night', 'scream', 'persistence_of_memory']
num_images = len(image_names)

# Create figure and axes
plt.close('all')
fig = plt.figure(figsize=(3*num_images+2, 8))
axs1 = []
axs2 = []
for i in range(num_images):
    ax1 = plt.subplot(2, num_images, i+1)
    ax2 = plt.subplot(2, num_images, i+1+num_images, polar=True)
    axs1.append(ax1)
    axs2.append(ax2)

# Open images and analyze
for ax1, ax2, image_name in zip(axs1, axs2, image_names):
    file_path = IMAGE_DIR_PATH / f"{image_name}.jpg"
    sn = Image.open(file_path)
    base_colors = np.array(sn)
    raw_colors = base_colors.reshape(-1, 3)
    raw_colors = np.unique(raw_colors, axis=0)
    rgb_colors = raw_colors/255.0
    hsv_colors = rgb_to_hsv(rgb_colors)
    hue, sat, val = [hsv_colors[:, i] for i in (0, 1, 2)]
    angle = 2*np.pi*hue
    radius = sat*val

    ax1.imshow(base_colors)
    ax1.axis('off')
    ax1.set_title(image_name)

    ax2.scatter(angle, radius, c=rgb_colors, s=4, edgecolors='none')
    ax2.grid(True)
    ax2.axes.yaxis.set_visible(False)

fig.tight_layout()
plt.show()
