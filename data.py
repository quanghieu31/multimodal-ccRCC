import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet18_Weights

import numpy as np

import configparser
config = configparser.ConfigParser()
config.read("config.ini")


class PatientClinicalDataset(Dataset):
    pass

class PatientRNASeqDataset(Dataset):
    pass

class PatientWSIDataset(Dataset):
    """
    dataset for accessing a patient's list of patches features, each is of shape (1, n_patches, n_features)
    """
    def __init__(self, 
        wsi_dir=config["wsi"]["wsi_slides"], 
        labels_dir=config["labels"]["labels"]
    ):
        self.wsi_dir = wsi_dir
        labels_df = pd.read_csv(labels_dir)
        self.case_ids = list(labels_df["submitter_id"].values)
        self.events = list(labels_df["event"].values)
        self.times = list(labels_df["time"].values)
        self.wsi_dirs = [os.path.join(self.wsi_dir, c) for c in self.case_ids]

    def __getitem__(self, idx):
        case_dir = self.wsi_dirs[idx]
        patches_features = np.load(case_dir + "/patches_features.npy", allow_pickle=True).item()
        cluster_ids = self.clustering(patches_features)

        features_list = list(patches_features.values())
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        phenotype_tensors = [] # list of tensors, each tensor is a cluster's features of shape i.e. (1, 15 patches in this cluster, 512 as output of resnet18)

        for cluster in unique_clusters:
            cluster_features = [features for features, c in zip(features_list, cluster_ids) if c == cluster]
            tensor_cluster_features = torch.from_numpy(np.array(cluster_features)).float().unsqueeze(0) # (1, n_patches, n_features)

            phenotype_tensors.append(tensor_cluster_features.to(device))

        return phenotype_tensors, torch.tensor(self.times[idx], dtype=torch.float32), torch.tensor(self.events[idx], dtype=torch.float32)

    def clustering(self, patches_features, n_clusters=5):
        feature_vectors = list(patches_features.values())
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=50)
        cluster_ids = kmeans.fit_predict(feature_vectors)
        return cluster_ids

    def __len__(self):
        return len(self.case_ids)



## Fusion Multimodal

class MultimodalDataset(Dataset):
    """
    takes three data paths (clinical, rna-seq, histopath images)
    build a data out of 'em
    """
    def __init__(self, 
        clinical_data_path, 
        rna_seq_data_path, 
        wsi_data_path
    ):
        pass

    def __getitem__(self, idx):
        pass