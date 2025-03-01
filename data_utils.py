import torch
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import configparser
config = configparser.ConfigParser()
config.read("config.ini")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PatientClinicalDataset(Dataset):
    """
    from csv, so getitem would be something like .loc[idx]
    """
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.df = pd.read_csv(self.csv_file_path).drop(["time", "event"], axis=1)

    def __getitem__(self, idx):
        patient_series = self.df.iloc[idx]
        return patient_series # includes the submitter_id!

    def __len__(self):
        return self.df.shape[0]


class PatientRNASeqDataset(Dataset):
    """
    a csv file, 534 rows and ~20000 columns for normalized RNA-seq counts
    """
    def __init__(self, rna_file_path):
        self.rna_file_path = rna_file_path
        self.df = pd.read_csv(self.rna_file_path)
        self.df.set_index("submitter_id", inplace=True)

    def __getitem__(self, case_id):
        gene_expressions = list(self.df.loc[case_id])
        tensor_gene_expressions = torch.tensor(gene_expressions, dtype=torch.float32).unsqueeze(0)
        return tensor_gene_expressions # [1, 19962]


class PatientWSIDataset(Dataset):
    """
    dataset for accessing a patient's list of patches features, each is of shape (1, n_patches, n_features)
    """
    def __init__(self, wsi_dir):

        self.wsi_dir = wsi_dir
        self.case_ids = list(os.listdir(self.wsi_dir))
        self.dict_case_id_path = {
            c: os.path.join(self.wsi_dir, c) + "/patches_features.npy" for c in self.case_ids
        }

    def __getitem__(self, case_id):
        # grab the list of 5 clusters for this case_id

        case_npy_file = self.dict_case_id_path[case_id]
        patches_features = np.load(case_npy_file, allow_pickle=True).item()
        
        cluster_ids = self.clustering(patches_features)

        features_list = list(patches_features.values())
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        list_phenotype_tensors = [] # list of tensors, each tensor is a cluster's features of shape i.e. (1, 15 patches in this cluster, 512 as output of resnet18)

        for cluster in unique_clusters:
            cluster_features = [features for features, c in zip(features_list, cluster_ids) if c == cluster]
            tensor_cluster_features = torch.from_numpy(np.array(cluster_features)).float().unsqueeze(0) # (1, n_patches, n_features)

            list_phenotype_tensors.append(tensor_cluster_features.to(device))

        return list_phenotype_tensors # [t1,t2,t3,t4,t5]

    def clustering(self, patches_features, n_clusters=5):
        feature_vectors = list(patches_features.values())
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=50)
        cluster_ids = kmeans.fit_predict(feature_vectors)
        return cluster_ids

    def __len__(self):
        return len(self.case_ids)



## Fusion multimodal

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
        # prepare labels from the clinical data path
        self.LABELS_DF = pd.read_csv(clinical_data_path)[["submitter_id", "event", "time"]]
        # then by initializing the clinical_dataset, remove the time and event from the clinical features:
        self.clinical_dataset = PatientClinicalDataset(clinical_data_path)

        # initialize the datasets for each modality
        self.wsi_dataset = PatientWSIDataset(wsi_data_path)
        self.rna_dataset = PatientRNASeqDataset(rna_seq_data_path)

        # label dictionary with key=submitter_id and value=(event,time) for easy lookup
        self.labels_dict = {}
        for submitter_id, event, time in zip(self.LABELS_DF["submitter_id"], self.LABELS_DF["event"], self.LABELS_DF["time"]):
            self.labels_dict[submitter_id] = {"event": event, "time": time}

    def __len__(self):
        return len(self.clinical_dataset)

    def __getitem__(self, idx):

        # (1) start from clinical dataset
        patient_series = self.clinical_dataset[idx]
        case_id = patient_series["submitter_id"]
        clinical_features = list(patient_series.drop(["submitter_id"]))
        tensor_clinical_features = torch.tensor(clinical_features, dtype=torch.float32).unsqueeze(0) 
        # above: add batch dim (1, 13) instead of (13)

        # (2) grab the tensor for 20000 (processed) gene counts for that case id
        tensor_rna_genes = self.rna_dataset[case_id] # (1, 19962)

        # (2.5) NOTE: to save time for this moment, I will concat the clinical and rna together 
        # and build one feed-forward for the combined
        tensor_clinical_rna = torch.cat((tensor_clinical_features, tensor_rna_genes), dim=1) # (1, 19975)

        # (3) collect the list of phenotype tensor for that case id
        list_of_phenotype_tensors = self.wsi_dataset[case_id]

        # (4) labels
        time = self.labels_dict[case_id]["time"]
        event = self.labels_dict[case_id]["event"]

        return (
            tensor_clinical_rna,
            list_of_phenotype_tensors,
            time,
            event
        )

        # return (
        #     tensor_clinical_features, 
        #     tensor_rna_genes, 
        #     list_of_phenotype_tensors,
        #     time,
        #     event
        # )

