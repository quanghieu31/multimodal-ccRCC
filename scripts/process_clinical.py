from tcia_utils import nbia
import sys
# !{sys.executable} -m pip install --upgrade -q tcia_utils
import configparser
config = configparser.ConfigParser()
config.read("../config.ini")
import json, requests
import urllib
from io import StringIO
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# BEWARE: the GDC APIs can change the data formats
# Make sure to either use the API or manually download the TCGA-KIRC's clinical json file from https://portal.gdc.cancer.gov/
# the file should be named something like ""clinical.cohort.2025-02-26.json""

SELECTED_COLS = [
    "tumor_grade_last_diagnosis", # this and stage are less correlated with each other
    'ajcc_stage_last_diagnosis', 
    #'ajcc_t_tumorsize_last_diagnosis',
    'submitter_id', 'gender', 'age_at_last_diagnosis', 'event', 'time'
]

CASES_WITH_RNA_DATA = set(pd.read_csv(config["rna"]["rna_metadata"])["case_id"].values)

def process_one_item(one_item):

    # four cases don't have rna-seq data, ignore these cases
    if one_item["submitter_id"] not in CASES_WITH_RNA_DATA:
        return None

    # collect stages and ages at last diagnosis
    diags_list = []
    for diag in one_item["diagnoses"]:
        if diag.get("tissue_or_organ_of_origin", None) == "Kidney, NOS" and "ajcc_pathologic_stage" in diag:
            diags_list.append({
                "ajcc_stage_last_diagnosis": diag.get("ajcc_pathologic_stage", "unknown_ajcc_stage"),
                "age_at_last_diagnosis": diag.get("age_at_diagnosis", one_item["demographic"]["age_at_index"] * 365),
                "tumor_grade_last_diagnosis": diag.get("tumor_grade", "unknown_tumor_grade"), 
                "ajcc_n_lymph_last_diagnosis": diag.get("ajcc_pathologic_n", "ajcc_n_lymph_unknown"), 
                "ajcc_m_metastasis_last_diagnosis": diag.get("ajcc_pathologic_m", "unknown_ajcc_m_metastasis"), 
                "ajcc_t_tumorsize_last_diagnosis": diag.get("ajcc_pathologic_t", "unknown_ajcc_t_tumorsize")
            })
        elif diag.get("tissue_or_organ_of_origin", None) == "Kidney, NOS":
            diags_list.append({
                "ajcc_stage_last_diagnosis": diag.get("ajcc_pathologic_stage", "unknown_ajcc_stage"),
                "age_at_last_diagnosis": diag.get("age_at_diagnosis", one_item["demographic"]["age_at_index"] * 365),
                "tumor_grade_last_diagnosis": diag.get("tumor_grade", "unknown_tumor_grade"), 
                "ajcc_n_lymph_last_diagnosis": diag.get("ajcc_pathologic_n", "ajcc_n_lymph_unknown"), 
                "ajcc_m_metastasis_last_diagnosis": diag.get("ajcc_pathologic_m", "unknown_ajcc_m_metastasis"), 
                "ajcc_t_tumorsize_last_diagnosis": diag.get("ajcc_pathologic_t", "unknown_ajcc_t_tumorsize")
            })
    # only pick the stage and info of most recent age
    final_dict = sorted(diags_list, key=lambda x: x["age_at_last_diagnosis"], reverse=True)[0]

    # collect general and demo info
    final_dict["submitter_id"] = one_item["submitter_id"]
    final_dict["gender"] = one_item["demographic"]["gender"]
    final_dict["race"] = one_item["demographic"]["race"]
    final_dict["event"] = one_item["demographic"]["vital_status"]
    
    if final_dict["event"] == "Dead":
        time = int(one_item["demographic"]["days_to_death"])
    else:
        for followup in one_item["follow_ups"]:
            if followup.get("timepoint_category", None) == 'Last Contact':
                time = followup["days_to_follow_up"]
    final_dict["time"] = time

    return final_dict


def clean(df):
    
    # TODO: should i remove the 3 samples with unknown staging

    df = df[SELECTED_COLS]  
    # convert event to binary nums: Dead=1, Alive=0
    df.loc[:, "event"] = df["event"].map({"Dead": 1, "Alive": 0})
    df.loc[:, "gender"] = df["gender"].map({"male": 1, "female": 0})

    # standardize the age_diagnosis because it's normally distributed and has relatively large values compared to the other
    df["age_at_last_diagnosis"] = df["age_at_last_diagnosis"].astype("float")
    mean_age, std_age = df["age_at_last_diagnosis"].mean(), df["age_at_last_diagnosis"].std()
    df.loc[:, "age_at_last_diagnosis"] = ((df["age_at_last_diagnosis"] - mean_age) / std_age)

    # one-hot encoding the categorical variable
    cats = [col for col in df.columns if df[col].dtypes == "object" and col not in  ("submitter_id", "event", "gender")]
    df_encoded = pd.get_dummies(data=df, columns=cats, dtype=int)

    return df_encoded



if __name__ == "__main__":

    with open(config["clinical"]["raw_clinical_json"], 'r') as f:
        raw_json = json.load(f)

    # select features and convert to json
    print("selecting features and convert from json to csv file...")
    list_dicts = [process_one_item(item) for item in raw_json if process_one_item(item)]
    df_json = pd.DataFrame(list_dicts)
    df_json.to_csv(config["clinical"]["converted_to_clinical_json"], index=False)

    # process/clean features for modeling
    print("cleaning the features for modeling")
    df_json_cleaned = clean(df_json)
    df_json_cleaned.to_csv(config["clinical"]["cleaned_clinical_json"], index=False)

    print("final cleaned csv file exists for clinical data", os.path.exists(config["clinical"]["cleaned_clinical_json"]))