from tcia_utils import nbia
import sys
# !{sys.executable} -m pip install --upgrade -q tcia_utils
import configparser
config = configparser.ConfigParser()
config.read("config.ini")
import json, requests
import urllib
from io import StringIO
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder



def queryBuilder(endpoint, filters, fields, size, frmat):
    api_url = 'https://api.gdc.cancer.gov/'
    if frmat.lower() == 'json':
        request_query = api_url + endpoint + '?filters=' + filters + '&fields=' + fields + '&size=' + size + '&format=' + frmat + '&pretty=true'
    else:
        request_query = api_url + endpoint + '?filters=' + filters + '&fields=' + fields + '&size=' + size + '&format=' + frmat
    return request_query


def get_raw_clinical():
    # step 1: get filter
    combination_two = {
                "op":"=",
                "content":{
                    "field": "cases.project.project_id", 
                    "value": "TCGA-KIRC"
        }
    }
    json_string=str(json.dumps(combination_two))
    filters_format = urllib.parse.quote(json_string.encode('utf-8'))
    #step 2: specify fields to be returned
    fields = ",".join([
        "submitter_id",
        "diagnoses.age_at_diagnosis", # instead of age_at_index
        "demographic.cause_of_death", "demographic.days_to_death",
        "demographic.gender", "demographic.race", "demographic.vital_status",
        "diagnoses.ajcc_pathologic_m", "diagnoses.ajcc_pathologic_n",
        "diagnoses.ajcc_pathologic_stage", "diagnoses.ajcc_pathologic_t", 
        "diagnoses.days_to_last_follow_up", 
    ])
    #step 3+4: specify size=2 and format=tsv, build query url with 'cases' endpoint
    max_observations = '537'
    template = queryBuilder('cases', filters_format, fields, max_observations, "tsv")
    # step 5: send request
    content = requests.get(template)

    data = StringIO(content.text)
    df = pd.read_csv(data, sep="\t")
    output_path = config["clinical_path"]["raw_clinical"]
    df.to_csv(output_path, index=False)

    print("Done querying and saving the raw clinical csv file")


def clean_metadata_clinical(file_name = config["clinical_path"]["raw_clinical"]):
    """ 
    - creates time and event columns
    - removes unnecessary columns
    - fills missing data for 2 columns (only missing 2 and 3 respectively)
    - renames
    """
    df = pd.read_csv(file_name)
    # create a time and an event feature
    df["time"] = df.apply(
        lambda row: row["demographic.days_to_death"]
        if row["demographic.vital_status"] == "Dead"
        else row["diagnoses.0.days_to_last_follow_up"],
        axis=1,)
    df["time"] = df["time"].astype("uint16") # save memory usage in range 0 to 65535
    df = df.rename(columns={"demographic.vital_status": "event"})
    df["event"] = df["event"].apply(lambda x: 1 if x=="Dead" else 0).astype("int8")
    # remove the other date columns
    df = df.drop(["demographic.days_to_death", "diagnoses.0.days_to_last_follow_up"], axis=1)
    # fill misisng data unknown category ===>> to fill in ordinal code as -1 hmmmm # TODO
    df = df.fillna("unknown")
    # rename cols
    df = df.rename(columns={
        "demographic.gender": "gender",
        "demographic.race": "race",
        "diagnoses.0.age_at_diagnosis": "age_diagnosis",
        "diagnoses.0.ajcc_pathologic_m": "ajcc_m", 
        "diagnoses.0.ajcc_pathologic_n": "ajcc_n",
        "diagnoses.0.ajcc_pathologic_stage": "ajcc_stage",
        "diagnoses.0.ajcc_pathologic_t": "ajcc_t"})
    # convert to integer
    df['age_diagnosis'] = pd.to_numeric(df['age_diagnosis'], errors='coerce') 
    df['age_diagnosis'] = df['age_diagnosis'].fillna(df['age_diagnosis'].mean()).astype(int)
    return df


def encode_categorical(df):
    # ordinal
    df = df.copy()
    def ordinal(sorted_values):
        dict_ordinal = {value: i for i, value in enumerate(sorted_values) if value != "unknown"}
        if "unknown" in sorted_values:
            dict_ordinal["unknown"] = -1
        return dict_ordinal
    for col in ["ajcc_m", "ajcc_n", "ajcc_stage", "ajcc_t"]:
        dict_ordinal = ordinal(sorted(df[col].unique()))
        df[col] = df[col].map(dict_ordinal).astype("int8")

    # non-ordinal
    label_encoders = {
        'gender': LabelEncoder(),
        'race': LabelEncoder(),
    }
    for col, encoder in label_encoders.items():
        df[col] = encoder.fit_transform(df[col])

    return df




if __name__ == "__main__":

    get_raw_clinical()
    df = encode_categorical(clean_metadata_clinical())
    
    output_clinical_cleaned = config["clinical_path"]["cleaned_clinical"]

    df.to_csv(output_clinical_cleaned, index=False)
    print(f"Successfully wrote {output_clinical_cleaned} file \nFile exists? {os.path.isfile(output_clinical_cleaned)}")
