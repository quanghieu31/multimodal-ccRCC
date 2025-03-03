import numpy as np
import pandas as pd 
import random
import os
import math

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout_ratio = 0.5

class WSI_FCN(nn.Module):
    """
    https://arxiv.org/abs/2009.11169
    fully convolutional network for WSI
    takes 1 phenotype tensor/cluster of shape (1, n_patches, 512)
    outputs a local representation of that phenotype tensor of shape (1, 64)
    why FCN? on the numerical vectors? 
        - utilize the kernel, and especially kernel_size=1 because we can't have kernel_size>1 for randomly picked patches from the histopathology slides
        - so why not a simple fully connected network (MLP)? it's because it requires inputs with fixed dimension and we have varying number of patches for each cluster
    also, note that a patch -> FCN -> (1,64) shape. So if we have 300 patches or (300,64) shape, we would use avgpooling and get (1,64) as the final output for that cluster
    """
    def __init__(self, in_features, out_features=64):
        super(WSI_FCN, self).__init__()
        # conv1d because we only have a tensor of shape (N, C, L) = (1, 512, i.e. 272)
        self.conv = nn.Conv1d(in_features, out_features, 
            kernel_size=1 # kernel size = 1 is extremely important because we only want to the a single patch to be learned, 
            # doing i.e. 3x3 is no use because the patches are picked randomly, so can't use spatial relationship here
        )
        self.relu = nn.ReLU()
        # adaptive avg pooling to get a local representation of the phenotype tensor
        # NOTE: adapative pooling from (64, 300 patches) to (64,1) as the final output of that cluster
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (1, n_patches, n_features)
        # permute to (1, n_features, n_patches) so that n_features become channels why? because tensor in pytorch reads () https://stackoverflow.com/questions/51541532/which-part-of-pytorch-tensor-represents-channels
        # n_patches is the length of the sequence. why?
        # FYI: for a conv2D, input should be in (N, C, H, W) format. N is the number of samples/batch_size. C is the channels. H and W are height and width resp: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d 
        # but here we have conv1d: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d

        x = x.permute(0, 2, 1) # (1, 512, 300 patches)
        x = self.conv(x) # (1, 64, 300 patches)
        x = self.relu(x) # (1, 64, 300 patches)
        x = self.pool(x) # (1, 64, 1)
        x = x.view(x.size()[0], -1) # (1, 64)
        return x # (1, 64)


class WSI_Attention(nn.Module):
    """
    https://arxiv.org/abs/2009.11169 
    pooling attention mechanism for WSI
    takes a local representation of the phenotype tensor of shape (5, 64) in which 5 is the number of clusters
    outputs a global representation of the phenotype tensors of shape (64-dim) which is a weighted sum across 5 clusters for 64 features
        each case has a global representation
    """
    def __init__(self, in_features, out_features=64):
        super(WSI_Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh(),  # tanh because we want to normalize the weights
            # why tanh() >> output values in range (-1,1), allowing both neg and pos values, often used in attention scores
            nn.Linear(out_features, 1),
        )

    def forward(self, x):
        # apply softmax because we have different number of clusters for each case
        # x: (5, 64) stack representation of 5 clusters/phenotypes
        scores = self.attention(x) # (5, 1)
        att_weights = torch.softmax(scores, dim=0).T # (1,5) which is probabilities
        # weighted sum across the 5 clusters:
        weights_applied = att_weights @ x  # (5, 64) = (1,5) @ (5,64)
        # weighted_sum_vector = torch.sum(weights_applied, dim=0) # (1, 64) or (64)
        return weights_applied, att_weights


class Clinical_RNA_FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim=32, dropout_ratio=dropout_ratio):
        # https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
        # https://arxiv.org/pdf/1207.0580
        # For fully connected layers, dropout in all hidden layers works
        # better than dropout in only one hidden layer and more extreme probabilities tend to be worse,
        # which is why we have used 0.5 throughout this paper
        
        super(Clinical_RNA_FeedForward, self).__init__()

        # hidden = [512, 256, 256, 64, 64, 32]
        # hidden = [1024, 512, 512, 256, 256, 128, 128, 64, 64, 32]

        hidden = [1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64, 32] # final: march 2, 2025

        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout_ratio),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout_ratio),
            nn.Linear(hidden[1], hidden[2]), nn.ReLU(), nn.Dropout(dropout_ratio),  
            nn.Linear(hidden[2], hidden[3]), nn.ReLU(), nn.Dropout(dropout_ratio),
            nn.Linear(hidden[3], hidden[4]), nn.ReLU(), nn.Dropout(dropout_ratio),
            nn.Linear(hidden[4], hidden[5]), nn.ReLU(), nn.Dropout(dropout_ratio),
            nn.Linear(hidden[5], hidden[6]), nn.ReLU(), nn.Dropout(dropout_ratio),
            nn.Linear(hidden[6], hidden[7]), nn.ReLU(), nn.Dropout(dropout_ratio),
            nn.Linear(hidden[7], hidden[8]), nn.ReLU(), nn.Dropout(dropout_ratio),
            nn.Linear(hidden[8], hidden[9]), nn.ReLU(), nn.Dropout(dropout_ratio),
            nn.Linear(hidden[9], hidden[10]), nn.ReLU(), nn.Dropout(dropout_ratio),
            nn.Linear(hidden[10], output_dim), nn.ReLU(), nn.Dropout(dropout_ratio),    
        )

    def forward(self, x):
        out = self.feedforward(x)
        return out


# FusionFeedForward
class FusionNetwork(nn.Module):
    def __init__(self, 
        input_dim_clinical_rna=19975, 
        input_dim_wsi_fcn=512, 
        input_dim_wsi_attention=64, 
        input_dim_final=96  # as from 32+64 = (out_dim of clinical_RNA) + (out_dim of WSI)
    ): 
        # NOTE: no dropout for now
        super(FusionNetwork, self).__init__()

        # Clinical+RNA
        self.clinical_rna_feedforward = Clinical_RNA_FeedForward(input_dim_clinical_rna, output_dim=32, dropout_ratio=dropout_ratio)
        # WSI_FCN and WSI_Attention
        self.wsi_fcn = WSI_FCN(input_dim_wsi_fcn, out_features=64)
        self.attention = WSI_Attention(input_dim_wsi_attention, out_features=64)

        # after fusion:
        # TODO: rational -> book: many hidden neurons are good -> with regularization like dropout/weight decay
        # for no. layers -> background knowledge and experimentation, for now 4 layers
        hidden = [64, 32, 16, 8]

        self.baby_feed_forward = nn.Sequential(
            nn.Linear(input_dim_final, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], hidden[2]),
            nn.ReLU(),
            nn.Linear(hidden[2], hidden[3]),
            nn.ReLU(),
            nn.Linear(hidden[3], 1)
        )

    def forward(self, tensor_clinical_rna, list_of_phenotype_tensors):

        # Clinical+RNA:
        extracted_clinical_rna = self.clinical_rna_feedforward(tensor_clinical_rna.to(device)) # (1, 32) shape
        
        # WSI_FCN
        local_reps = [] # len=5
        # here since tensors have different no. images in each of them
        # we use the "flexiblity" of the FCN to output 1x64 for each cluster
        for tensor in list_of_phenotype_tensors:
            tensor = tensor.to(device)
            cluster_rep = self.wsi_fcn(tensor) # each of shape (1,64) by pooling from tensors with varying dim
            local_reps.append(cluster_rep)
        # stack 5 local representation of shape (1,64) >> tensor of shape (5,64)
        tensor_local_reps = torch.cat(local_reps)

        # WSI_Attention:
        wsi_aggregated_vector, att_weights = self.attention(tensor_local_reps) # from (5,64) to weighted vector (1,64)

        # concantenate:
        concatenated_features = torch.cat((extracted_clinical_rna, wsi_aggregated_vector), dim=1) # shape (1, 96)

        risk_score = self.baby_feed_forward(concatenated_features)
        return risk_score
