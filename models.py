import numpy as np
import pandas as pd 
import random
import os

import torch
from torch.utils.data import Dataset, DataLoader, random_split

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def negative_partial_log_likelihood(hazard_pred, time, event, device, eps=1e-8):
    # flatten predictions
    hazard_pred = hazard_pred.view(-1)
    time = time.to(device, dtype=torch.float).view(-1)
    event = event.to(device, dtype=torch.float).view(-1)

    if event.sum() == 0:
        return torch.tensor(0.0, device=device)

    # compute risk set: R[i, j] = 1 if time[j] >= time[i]
    R_mat = torch.tensor(np.greater_equal.outer(time.cpu(), time.cpu()).T.astype(np.float32), device=device)

    # standardize theta
    theta = (hazard_pred - hazard_pred.mean()) / (hazard_pred.std(unbiased=False) + eps)

    # compute the log risk set using the correct formula
    # NOTE: use theta directly without an extra exp()
    # First, mask the non-risk set entries by multiplying exp(theta) with R_mat,
    # then take the log of the sum
    log_risk_set = torch.log(torch.sum(torch.exp(theta) * R_mat, dim=1) + eps)

    # negative partial likelihood only for events
    loss = -torch.mean((theta - log_risk_set) * event)

    return loss

# one-batch
# hazard_pred = torch.tensor([2.1, 1.8, 3.0, 0.5, 2.5], device=device)
# time = torch.tensor([5, 3, 6, 2, 4], device=device)
# event = torch.tensor([1, 1, 0, 1, 0], device=device)

# one_batch_loss = negative_partial_log_likelihood(hazard_pred, time, event, device)
# print(one_batch_loss)


