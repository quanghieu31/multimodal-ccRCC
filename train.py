import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from datetime import datetime, date
from lifelines.utils import concordance_index
from utils import display_km_curves_fusion

from models import *
from data_utils import *

import configparser
config = configparser.ConfigParser()
config.read("config.ini")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

########################## LOSS ###############################################

def negative_partial_log_likelihood(hazard_preds, times, events, device, eps=1e-8):

    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    # flatten predictions

    hazard_preds = hazard_preds.view(-1)
    times = times.to(device, dtype=torch.float).view(-1)
    events = events.to(device, dtype=torch.float).view(-1)

    if events.sum() == 0:
        return torch.tensor(0.0, device=device)

    # compute risk set: R[i, j] = 1 if times[j] >= times[i]
    # https://stackoverflow.com/questions/56646261/can-someone-please-explain-np-less-equal-outerrange1-18-range1-13
    R_mat = torch.tensor(
        np.greater_equal.outer(times.cpu(), times.cpu()).T.astype(np.float32), 
        device=device
    )

    # standardize theta/hazard prediction
    theta = (hazard_preds - hazard_preds.mean()) / (hazard_preds.std(unbiased=False) + eps)

    # compute the log risk set using the correct formula
    # NOTE: use theta directly without an extra exp()
    # First, mask the non-risk set entries by multiplying exp(theta) with R_mat,
    # then take the log of the sum
    log_risk_set = torch.log(torch.sum(torch.exp(theta) * R_mat, dim=1) + eps)

    # negative partial likelihood only for events
    loss = -torch.mean((theta - log_risk_set) * events)

    return loss

# one-batch
# hazard_pred = torch.tensor([2.1, 1.8, 3.0, 0.5, 2.5], device=device)
# time = torch.tensor([5, 3, 6, 2, 4], device=device)
# event = torch.tensor([1, 1, 0, 1, 0], device=device)

# one_batch_loss = negative_partial_log_likelihood(hazard_pred, time, event, device)
# print(one_batch_loss)

################## HYPERPARAMS ################################################

# BEST OF NOW: March 4, 2025
# https://arxiv.org/pdf/1206.5533 (guide to choose hyperparams)
n_epochs = 5
lr = 0.0001
batch_size = 32

# regularizations:
dropout_ratio = 0.5
weight_decay = 0.0001

# since we have relatively small dataset (~300 for training), high weidght decay may lead to udnerfitting
# but we might have many interactions between parameters in the final feedforward, so let's try different ones
# https://medium.com/towards-data-science/this-thing-called-weight-decay-a7cd4bcfccab
# https://stackoverflow.com/questions/44452571/what-is-the-proper-way-to-weight-decay-for-adam-optimizer

################### TRAIN #####################################################

def custom_collate(batch):
    """
    https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders 
    i.e. 32 batch_size
    batch = [
        (list_of_phenotype_tensors, time1, event1),    => case 1
        (list_of_phenotype_tensors, time2, event2),    => case 2
        ...                                            => case 32
    ]

    TODO: more explanation to come
    """
    # each element in batch is a tuple: 
    # (clinical_rna_tensor, list_of_phenotype_tensors, time, event)
    list_of_clinical_rna_features, list_of_lists_of_5_tensors, times, events = zip(*batch)
    # i.e. [patient_1_clinical_rna_t1, patient_2_clinical_rna_t2,..., patient32_clinical_rna_t32]
    # i.e. [[t1,t2,t3,t4,t5],[t1,t2,t3,t4,t5],...,[t1,t2,t3,t4,t5]] = [32 lists]
    
    return (
        list(list_of_clinical_rna_features),
        list(list_of_lists_of_5_tensors),
        torch.tensor(times),
        torch.tensor(events)
    )

# dataset and splitting
dataset = MultimodalDataset(
    config["clinical"]["cleaned_clinical_json"],
    config["rna"]["cleaned_rna"],
    config["wsi"]["wsi_slides"]
)

train_size, val_size = int(0.7 * len(dataset)), int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train, val, test = random_split(dataset, [train_size, val_size, test_size])

# loading data
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val, batch_size=val_size, shuffle=False, collate_fn=custom_collate)
test_loader = DataLoader(test, batch_size=test_size, shuffle=False, collate_fn=custom_collate)

# initiate model
model = FusionNetwork(
    input_dim_clinical_rna=19975,
    input_dim_wsi_fcn=512,
    input_dim_wsi_attention=64,
    input_dim_final=96
)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def train():
    print("begin to train")
    # training loops
    for epoch in range(n_epochs):

        model.train()
        train_loss = 0.0 

        # each batch contains 32 cases!
        for batch in tqdm(train_loader):
            batch_clinical_rna_features, batch_lists_phenotype_clusters, batch_times, batch_events = batch

            risk_scores = [] # list of  32 risk scores
            for (clinical_rna_features, list_of_phenotype_tensors) in zip(batch_clinical_rna_features, batch_lists_phenotype_clusters):
                # process each sample in the batch of 32
                risk_score = model(clinical_rna_features, list_of_phenotype_tensors)
                risk_scores.append(risk_score)

            # convert to tensor type
            risk_scores = torch.stack(risk_scores) # of shape (batch_size, 1) or (32,1) 

            # TODO: explain in detail: meaning of loss of 32 cases in the batch
            optimizer.zero_grad() # zero the parameter gradients
            loss = negative_partial_log_likelihood(risk_scores, batch_times.to(device), batch_events.to(device), device)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"epoch {epoch}, loss: {train_loss / len(train_loader)}")

    print("finished training\n")


    ####################### VALIDATION ############################################

    print("begin to validate")
    model.eval()

    val_risks = []
    val_times = []
    val_events = []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            # unpack the batch
            batch_clinical_rna_features, batch_lists_phenotype_clusters, batch_times, batch_events = batch
            
            # move times and events to the device
            batch_times = batch_times.to(device)
            batch_events = batch_events.to(device)
            
            # tterate over each sample in the batch
            for idx, (clinical_rna_features, list_of_phenotype_tensors) in enumerate(zip(batch_clinical_rna_features, batch_lists_phenotype_clusters)):
                
                risk_score = model(clinical_rna_features, list_of_phenotype_tensors)
                
                val_risks.append(risk_score.item())
                val_times.append(batch_times[idx].item())
                val_events.append(batch_events[idx].item())

    val_c_index = concordance_index(val_times, -np.array(val_risks), val_events)
    print(f"validation c-index: {val_c_index}")

    display_km_curves_fusion(val_risks, val_times, val_events, "validation set", save_figure=True)

    saved_model = True
    if saved_model:
        checkpoint_path = f"checkpoints/trained-model_{date.today()}_{val_c_index:4f}.pth"
        torch.save({
            'model_state_dict': model.state_dict(), # all weights all models
            'optimizer_state_dict': optimizer.state_dict(),
            'batch_size': batch_size,
            'dropout_ratio': dropout_ratio,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'n_epochs': n_epochs,
            'random_seed': 0,
            'val_c_index': val_c_index
        }, checkpoint_path)
        print(f"saved model: {checkpoint_path}")


if __name__ == "__main__":
    train()