import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import csv

from lifelines.utils import concordance_index
from utils import display_km_curves_fusion

from models import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################# FINAL ###########################################

checkpoint_path = _ # TODO: to choose

model_chkpt = FusionNetwork()
model_chkpt.to(device)
optimizer = optim.Adam(model_chkpt.parameters())

# load from last check point
checkpoint = torch.load(checkpoint_path, map_location=device)
model_chkpt.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
val_c_index = checkpoint['val_c_index']

model_chkpt.eval()

test_risks = []
test_times = []
test_events = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        # unpack the batch
        batch_clinical_rna_features, batch_lists_phenotype_clusters, batch_times, batch_events = batch
        
        # move times and events to the device
        batch_times = batch_times.to(device)
        batch_events = batch_events.to(device)
        
        # tterate over each sample in the batch
        for i, (clinical_rna_features, list_of_phenotype_tensors) in enumerate(zip(batch_clinical_rna_features, batch_lists_phenotype_clusters)):
            
            risk_score = model_chkpt(clinical_rna_features, list_of_phenotype_tensors)
            
            test_risks.append(risk_score.item())
            test_times.append(batch_times[i].item())
            test_events.append(batch_events[i].item())

test_c_index = concordance_index(test_times, -test_risks, test_events)
print(f"test c-index: {test_c_index}")

display_km_curves_fusion(test_risks, test_times, test_events, "test set", save_figure=True)

# append to csv file
with open("evaluation-results/c-index-results.csv", "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "c-index"])
    writer.writerow({"model": "fusion_network", "c-index": test_c_index})
