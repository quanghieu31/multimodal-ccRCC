import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
import re
from collections import defaultdict
import random
from concurrent.futures import ThreadPoolExecutor
import math
from sklearn.cluster import KMeans
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
from torchvision.models import VGG19_Weights, ResNet18_Weights
from torchvision import models

import utils



np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

num_patches = 1000  # arbitrary, #TODO
patch_size = 500   # patch size at 20X


# this loop iterates over the SVS files in the examples_2_patient folder
# and then extracts patches from the tissue mask of each slide

def extract_patches_from_tissue_mask(subject_svs_filepath, num_patches=num_patches, patch_size=patch_size):

    patches_folder = os.path.dirname(subject_svs_filepath) + f"/patches"
    Path(patches_folder).mkdir(parents=True, exist_ok=True)
    print(f"\textracting patches from {subject_svs_filepath}")

    if len(os.listdir(patches_folder)) == num_patches * (len(os.listdir(os.path.dirname(subject_svs_filepath)))-1): # -1 for the patches folder
        return

    svs_file_id = subject_svs_filepath.split('/')[-1][:-4]

    slide = openslide.OpenSlide(subject_svs_filepath)

    level = 2   #TODO be careful, one of the slides crashed at level 1

    thumbnail = slide.read_region((0, 0), level, slide.level_dimensions[level])
    thumbnail = thumbnail.convert("RGB") # convert from PIL format to RGB
    thumbnail_np = np.array(thumbnail) # convert to numpy array

    #### apply otus thresholding to extract tissue mask
    gray = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2GRAY)
    _, tissue_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # invert mask so tissue = white (255), background = black (0)
    tissue_mask = cv2.bitwise_not(tissue_mask)

    #############################################  
    #### zooming patches to the tissue mask #####
    #############################################

    # use scale factor instead of resizing (avoid mem error)
    scale_factor = slide.level_downsamples[level]  
    mask_coords = np.column_stack(np.where(tissue_mask > 0))  # (y, x) positions at level 2

    # map mask coordinates to level 0 size (20X magnification)
    scaled_coords = [(int(y * scale_factor), int(x * scale_factor)) for y, x in mask_coords]
    # print("\t-- scaled_coords, first 5 coords:", scaled_coords[:5])  

    # randomly sample coordinates for patches
    sampled_coords = random.sample(scaled_coords, min(num_patches, len(scaled_coords)))

    # collect/save patches
    for i, (y, x) in enumerate(sampled_coords): #, desc="extracting patches"):
        # read the patch from the slide
        patch = slide.read_region((x, y), 0, (patch_size, patch_size))  
        patch = patch.convert("RGB")   # convert from PIL format to RGB
        patch_filename = f"{patches_folder}/{svs_file_id}_{y}_{x}.png"
        patch.save(patch_filename, "PNG")

    slide.close()

    print(f"\tsaved {len(os.listdir(patches_folder))} patches in {patches_folder}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

#############################################
### data loader for the extracted patches ###
#############################################
class PatchDataset(Dataset):
    """
    Patches
    """
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = [os.path.join(self.folder, f) for f in os.listdir(folder)
                      if os.path.isfile(os.path.join(self.folder, f))]
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        # convert from BGR to RGB
        image = cv2.imread(self.files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        # return image object and filename
        filename = self.files[idx].split("\\")[-1]
        return image, filename


#############################################
### define transforms for VGG19 input #######
#############################################
# VGG19 expects 224x224 images and imagnet normalization
# and ResNet18 augmentations # NOTE
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])


#############################################
####### load pre-trained VGG19 model ########
#############################################
#############################################
### extract features for each patch ##########
#############################################
# want to extract a feature vector rather than classification scores.
# thus, build a new Sequential model that goes through
# 1. model.features (convolutional part)
# 2. model.avgpool (adaptive avg pooling)
# 3. flatten (convert a tensor to a vector)
# 4. model.classifier = False 
# result is a some-dimensional feature vector for each patch image

def extract_features_from_all_patches(dataloader, model_name="resnet18", device=device):
    """
    extracts features from patches of one SVS file
    """
    feature_extractor = None

    if model_name == "vgg19":
        # pretrained imagenet
        model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        model.to(device)
        model.eval()   # no need to train, use the current weights to extract features

        feature_extractor = torch.nn.Sequential(
            model.features,
            model.avgpool,
            torch.nn.Flatten(),
        *list(model.classifier.children())[:-1] # remove the last layer
    ).to(device)

    elif model_name == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.to(device)
        model.eval()   # no need to train, use the current weights to extract features

        feature_extractor = torch.nn.Sequential(
            *list(model.children())[:-1],  # all layers except the final fc layer
            torch.nn.Flatten()
        ).to(device)

    patches_features = {}

    with torch.no_grad():
        for image, file_name in tqdm(dataloader):
            image = image.to(device)
            features_model = feature_extractor(image)  
            # to numpy array
            patches_features[file_name[0]] = features_model.squeeze(0).cpu().numpy()  

    return patches_features



###############################################################################
###############################################################################


if __name__ == "__main__":

    # NOTE CONFIGS:

    with open(config["case_ids"]["full_case_ids"], "r") as file:
        full_case_ids = [line.strip() for line in file]

    current = os.listdir(r"D:\data\wsi_slides")
    undone_case_ids = [c for c in full_case_ids if c not in current]


    selected_cases_to_download = undone_case_ids[0:20] # TODO: just run this immediately
    

    cases_with_big_tumors = [] # visual later, need to find some # TODO to save the original svs files and patch pngs
    cases_without_slides = []
    num_slides = 5


    for case in selected_cases_to_download:

        case_dir = config["wsi"]["wsi_slides"] + case
        os.makedirs(case_dir, exist_ok=True)

        if os.path.exists(f"{case_dir}/patches_features.npy"):
            continue

        # download the slides
        try:
            utils.download_slides(case, case_dir, num_slides=num_slides)
            if len(os.listdir(case_dir)) not in range(1, num_slides+1):
                cases_without_slides.append(case)
                continue
        except Exception as e:
            print(e)
            continue
        print(f"{case}'s {len(os.listdir(case_dir))} slides are downloaded")

        # extract patches from tissue mask from each svs slide
        svs_paths = [f"{case_dir}/{svs_file}" for svs_file in os.listdir(case_dir) if svs_file.endswith('.svs')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(extract_patches_from_tissue_mask, svs_paths)

        # after getting the patches, delete all the SVS files
        # to save space # NOTE: very important
        for svs_file in svs_paths:
            os.remove(svs_file)

        # extracting features from all patches (from all svs slides) of this case
        patches_folder = f"{case_dir}/patches"
        dataset = PatchDataset(patches_folder, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        print(f"\textracting features from {len(dataset)} patches")
        patches_features = extract_features_from_all_patches(dataloader, model_name="resnet18", device=device)

        # save the features to a npy file
        np.save(f"{case_dir}/patches_features.npy", patches_features)

        # delete the patches folder to save space
        shutil.rmtree(patches_folder)

    print("cases without slides:", cases_without_slides)

    # # NOTE: to delete
    # # visual purpose for this cohort
    # # write the case ids to a file
    # with open("case_ids_80_90_visual.txt", "w") as file:
    #     for case in cases_with_big_tumors:
    #         file.write(case + "\n")
