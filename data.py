import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet18_Weights

import numpy as np
import pandas as pd

import configparser
config = configparser.ConfigParser()
config.read("config.ini")


class PatientClinicalDataset(Dataset):
    """
    from csv, so getitem would be something like .loc[idx]
    """
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.df = pd.read_csv(self.csv_file_path).drop(["time", "event"], axis=1)

    def __getitem__(self, idx):
        patient = self.df.iloc[idx]
        return patient

    def __len__(self):
        return self.df.shape[0]



class PatientRNASeqDataset(Dataset):
    """
    currently still in tsv file, might want to think of better way?
    """
    # TODO




class PatientWSIDataset(Dataset):
    """
    dataset for accessing a patient's list of patches features, each is of shape (1, n_patches, n_features)
    """
    def __init__(self, wsi_dir):

        self.wsi_dir = wsi_dir
        self.case_ids = list(LABELS_DF["submitter_id"].values)
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


    def __getitem__(self, idx):

        patient_time = torch.tensor(self.times[idx])
        patient_event = torch.tensor(self.events[idx])

        # (1) start from clinical dataset
        patient = self.clinical_dataset[idx]
        case_id = patient["submitter_id"]
        clinical_features = list(patient.drop(["submitter_id"]))
        tensor_clinical_features = torch.tensor(clinical_features).unsqueeze(0) # add batch dim (1, 18) instead of (18)

        # (2) grab the tensor for 20000 (processed) gene counts for that case id
        tensor_rna_genes = self.rna_dataset[case_id]

        # (3) collect the list of phenotype tensor for that case id
        list_of_phenotype_clusters = self.wsi_dataset[case_id]

        return (tensor_clinical_features, tensor_rna_genes, list_of_phenotype_clusters,
                self.labels_dict[submitter_id]["event"], self.labels_dict[submitter_id]["time"] # labels
        )

