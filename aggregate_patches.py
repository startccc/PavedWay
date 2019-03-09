#%%

import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from dataclasses import dataclass
from PIL import Image
from typing import List

#%%
images_format = "/Users/jb/Documents/flathack/data/chauderon/*.png"

#%%
@dataclass
class Patch:
    lat: float
    lon: float
    values: np.array


#%%
# load all patches from directory

patches = []
for image_file in glob(images_format):
    lat, lon = list(map(float, image_file.split("/")[-1][: -len(".png")].split("_")[1:3]))
    img = Image.open(image_file)
    img.load()
    data = np.asarray(img, dtype="float")
    patches.append(Patch(lat, lon, data))

#%%
gps_width = 0.000_423_0
image_width = 1280
pixels_gps_ratio = float(image_width) / gps_width

# 6.622234 - 6.622666
# 6.621872 - 6.622301


def aggregate_patches(patches: List[Patch], aggregate: str = "superpose"):
    centers_geo = np.array([[p.lat, p.lon] for p in patches])
    print("centers_geo", centers_geo)
    patch_top_left_geo = centers_geo + (np.array([gps_width, -gps_width]) / 2)
    print("patch_top_left_geo", patch_top_left_geo)
    agg_top_left_geo = np.array([patch_top_left_geo[:, 0].max(), patch_top_left_geo[:, 1].min()])
    print("agg_top_left_geo", agg_top_left_geo)
    patch_top_left_pixels = (
        (patch_top_left_geo - agg_top_left_geo) * (image_width / gps_width) * np.array([-1, 1])
    ).astype(int)
    print("patch_top_left_pixels", patch_top_left_pixels)

    patch_bottom_right_pixels = patch_top_left_pixels + np.array([image_width, image_width])
    print("patch_bottom_right_pixels", patch_bottom_right_pixels)
    agg_bottom_right_pixels = np.array(
        [patch_bottom_right_pixels[:, 0].max(), patch_bottom_right_pixels[:, 1].max()]
    )
    print("agg_bottom_right_pixels", agg_bottom_right_pixels)

    height, width = agg_bottom_right_pixels
    aggregate_image = np.zeros([height, width])

    for i, patch in enumerate(patches):
        aggregate_image[
            patch_top_left_pixels[i, 0] : patch_top_left_pixels[i, 0] + image_width,
            patch_top_left_pixels[i, 1] : patch_top_left_pixels[i, 1] + image_width,
        ] = patch.values

    return aggregate_image


#%%
img = aggregate_patches(patches)
plt.figure(figsize=(10, 10))
plt.imshow(img)
