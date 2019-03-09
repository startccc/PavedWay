#%%

import requests
import os
from tqdm import tqdm
import shutil
import numpy as np

#%%

# PARAMETERS
output_directory = "/Users/jb/Documents/flathack/output"
os.makedirs(output_directory, exist_ok=True)
# Create ACCESS_KEY variable in interpreter
patch_stride = patch_width = 0.00025
scale = 2
zoom = 25
size = 2000

#%%
def bounding_box_coords_generator(
    lat_top_left, lon_top_left, lat_bottom_right, lon_bottom_right, patch_stride
):
    """iterators of coordinates of all patches in bounding box"""
    coords = []
    for lat in np.arange(
        lat_bottom_right + patch_stride / 2, lat_top_left - patch_stride / 2, patch_stride
    ):
        for lon in np.arange(
            lon_top_left + patch_stride / 2, lon_bottom_right - patch_stride / 2, patch_stride
        ):
            coords.append((lat, lon))
    return coords


def map_url_request(lat, lon, scale, zoom, size):
    """create URL for Google API"""
    return f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&scale={scale}&zoom={zoom}&size={size}x{size}&maptype=satellite&key={ACCESS_KEY}"


def download_map_image(url, filename):
    """download the image and save"""
    requests.get(url)
    response = requests.get(url, stream=False, allow_redirects=True)
    with open(filename, "wb") as out_file:
        out_file.write(response.content)
    return response


def download_patches_in_bounding_box(
    lat_top_left, lon_top_left, lat_bottom_right, lon_bottom_right, files_prefix
):
    """download all prefix in bounding box"""
    coords = bounding_box_coords_generator(
        lat_top_left, lon_top_left, lat_bottom_right, lon_bottom_right, patch_stride
    )
    if len(coords) > 100:
        raise ValueError(f"Too many patches to download ({len(coords)})")

    for lat, lon in tqdm(coords):
        out_filename = f"{files_prefix}_{lat:.6f}_{lon:.6f}.png"
        if not os.path.exists(out_filename):
            url = map_url_request(lat, lon, scale, zoom, size)
            download_map_image(url, out_filename)


#%%

if False:
    download_patches_in_bounding_box(
        46.525_656, 6.622_070, 46.523_226, 6.625_030, os.path.join(output_directory, "chauderon")
    )


#%%
files_prefix = os.path.join(output_directory, "image-charles")
with open("locations-charles", "r") as f:
    for line in f.readlines():
        print(line)
        line = line.strip()
        if not line or line[0] == "#":
            continue
        lat, lon = line.split(", ")
        lat = float(lat.strip())
        lon = float(lon.strip())
        out_filename = f"{files_prefix}_{lat:.6f}_{lon:.6f}.png"
        url = map_url_request(lat, lon, scale, zoom, size)
        download_map_image(url, out_filename)


#%%
