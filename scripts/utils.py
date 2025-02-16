import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import random

import json
import requests
import configparser
config = configparser.ConfigParser()
config.read("../config.ini")

import openslide
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import matplotlib.patches as patches
import math
from sklearn.cluster import KMeans


####################################################################################################
########################### WSI UTILS FUNCTIONS ####################################################
####################################################################################################

###### DOWNLOAD

# Function to query the GDC API for slide files associated with a submitter_id
def query_gdc_api(submitter_id):
    files_endpt = "https://api.gdc.cancer.gov/files"
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.submitter_id",
                    "value": [submitter_id]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_format",
                    "value": ["SVS"]
                }
            }
        ]
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name",
        "format": "JSON",
        "size": "100"
    }
    response = requests.get(files_endpt, params=params)
    if response.status_code == 200:
        return response.json()["data"]["hits"]
    else:
        print(f"Error querying GDC API: {response.status_code}")
        return []


# Function to download a file from the GDC API using its file_id
def download_slide(file_id, file_name, download_dir):
    data_endpt = f"https://api.gdc.cancer.gov/data/{file_id}"
    response = requests.get(data_endpt, stream=True)
    if response.status_code == 200:
        file_path = os.path.join(download_dir, file_name)
        with open(file_path, "wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    output_file.write(chunk)
        # print(f"Downloaded {file_name}")
    else:
        print(f"Error downloading {file_name}: {response.status_code}")


# Main function to execute the download process
def download_slides(submitter_id, download_dir, num_slides=10):
    # Ensure the download directory exists
    os.makedirs(download_dir, exist_ok=True)

    # Query the GDC API for slide files
    slides = query_gdc_api(submitter_id)

    if not slides:
        print("No slides found for the given submitter_id.")
        return

    # Randomly select the specified number of slides
    selected_slides = random.sample(slides, min(num_slides, len(slides)))

    # Download each selected slide
    for slide in selected_slides:
        download_slide(slide["file_id"], slide["file_name"], download_dir)


######## Displaying

def display_wsi(file_path, level=0):
    try:
        slide = openslide.OpenSlide(file_path)
        levels = slide.level_dimensions
        print(f"available levels and their dimensions: {levels}")
        if level >= len(levels):
            raise ValueError(f"level {level} is not valid. available levels: {len(levels)}.")

        # get the dimensions of the specified level
        dimensions = levels[level]
        print(f"image dimensions at level {level}: {dimensions}")

        # generate a thumbnail for easier visualization if the level is too large
        if dimensions[0] > 20000 or dimensions[1] > 20000:  # arbitrary threshold
            thumbnail = slide.get_thumbnail((1024, 1024))  # resize to 1024x1024
            plt.figure(figsize=(8, 8))
            plt.imshow(thumbnail)
            plt.axis("off")
            plt.title(f"thumbnail for {file_path.split('/')[-1]}")
            plt.show()
        else:
            # read the ROI at the specified level
            region = slide.read_region((0, 0), level, dimensions)
            plt.figure(figsize=(8, 8))
            plt.imshow(region)
            plt.axis("off")
            plt.title(f"{file_path.split('/')[-1]} (level {level})")
            plt.show()

    except Exception as e:
        print(f"error: {e}")

# for file_path in files_examples_2_patient:
#     display_wsi(file_path, level)

#############################################
##### Otsu's thresholding for tissue ########
##### views  ################################
#############################################

def display_tissue_extraction(file_path, level=2):
    slide = openslide.OpenSlide(file_path)
    print("available levels and their dimensions:", slide.level_dimensions)

    level = 2  # resolution level
    w, h = slide.level_dimensions[level]
    print(f"width and height at chosen level {level}:", w, h)

    # read the downsampled image
    thumbnail = slide.read_region((0, 0), level, (w, h))
    thumbnail = thumbnail.convert("RGB")  # convert from PIL format to RGB
    image_rgb = np.array(thumbnail) # convert to numpy array

    # Otsu's
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) # get the grayscale version
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # invert the binary mask (so tissue is white and background is black)
    binary_inv = cv2.bitwise_not(binary)

    # apply the mask to extract tissue while keeping original colors
    extracted_tissue = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_inv)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(image_rgb)  
    ax[0].set_title(f"thumbnail at level {level} - original")
    ax[0].axis("off")

    ax[1].imshow(gray, cmap="gray")
    ax[1].set_title("grayscale version")
    ax[1].axis("off")

    ax[2].imshow(binary_inv, cmap="gray")
    ax[2].set_title("otsu")
    ax[2].axis("off")

    ax[3].imshow(extracted_tissue)
    ax[3].set_title("extracted tissue, rgb")
    ax[3].axis("off")
    plt.show()


####### Clustering and display

def display_clustering(svs_path, p_features, n_clusters=5, level=2, patch_size=500):

    #### clustering ######
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=50)
    cluster_ids = kmeans.fit_predict(list(p_features.values()))
    patches_clusters = {(int(f.split("_")[1]), int(f.split("_")[2].split(".")[0])): c # (coord y, coord x): cluster
                        for f, c in zip(p_features.keys(), cluster_ids)}

    # display the slide with the patches colored by cluster
    slide = openslide.OpenSlide(svs_path) # TODO

    level_dims = slide.level_dimensions[level]  
    full_dims = slide.level_dimensions[0]        
    img = slide.read_region((0, 0), level, level_dims).convert("RGB")

    # compute scale factors from full resolution to the chosen level
    scale_x = level_dims[0] / full_dims[0]
    scale_y = level_dims[1] / full_dims[1]

    # colors for the clusters
    colors = ['red', 'blue', 'green', 'orange', 'purple', 
            'yellow', 'pink', 'cyan', 'magenta', 'lime']

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    for coord, cluster in patches_clusters.items():
        try:
            y_full, x_full = coord[0], coord[1]
        except Exception as e:
            print(f"skip coord {x_full, y_full} due to error: {e}")
            continue

        # scale the coordinates to the visualization level=2
        x_vis = x_full * scale_x
        y_vis = y_full * scale_y
        patch_size_vis = patch_size * scale_x  # isotropic scaling

        # 500x500 patch with the color of that cluster
        rect = patches.Rectangle(
            (x_vis, y_vis), patch_size_vis, patch_size_vis,
            linewidth=2,
            edgecolor=colors[cluster % len(colors)],  
            facecolor='none'
        )
        ax.add_patch(rect)

    plt.axis('off')
    plt.show()


####### Displaying both original and clustering

def display_cluster_and_original(patches_features):
    dict_svs_patches_features = defaultdict(dict)
    for f, c in patches_features.items():
        svs_file_path = example_2_patient_path + f.split("_")[0] + ".svs"
        dict_svs_patches_features[svs_file_path][f] = c

    print(dict_svs_patches_features.keys())

    for correspond_svs_path, pf in dict_svs_patches_features.items():
        print(f"processing {correspond_svs_path}")
        display_wsi(correspond_svs_path, level)
        display_clustering(correspond_svs_path, pf)
        print("==============================================================")
