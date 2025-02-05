'''
Coutersy of Steven Song - Teaching Assistant for CMSC 35440 Machine Learning in Biology and Medicine (Winter 2025) 
'''

import configparser
config = configparser.ConfigParser()
config.read("config.ini")

import json
import os
import re
import shutil
import tarfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import trange

max_size = 20000
batch_size = 10
files_endpt = "https://api.gdc.cancer.gov/files" # file metadata only
data_endpt = "https://api.gdc.cancer.gov/data" # actualy data download (files downloaded)
suffix = "rna_seq.augmented_star_gene_counts.tsv"

data_folder = config["genomics_data"]["rna_seq_count_path"]
os.makedirs(data_folder, exist_ok=True)

filters = {
    "op": "and",
    "content": [
        {
            "op": "in",
            "content": {
                "field": "cases.project.project_id",
                "value": ["TCGA-KIRC"],
            },
        },
        {
            "op": "in",
            "content": {
                "field": "files.experimental_strategy",
                "value": ["RNA-Seq"],
            },
        },
        {
            "op": "in",
            "content": {
                "field": "files.data_format",
                "value": ["tsv"],
            },
        },
        {
            "op": "in",
            "content": {
                "field": "files.data_category",
                "value": ["transcriptome profiling"],
            },
        },
        {
            "op": "in",
            "content": {
                "field": "files.data_type",
                "value": ["Gene Expression Quantification"],
            },
        },
        {
            "op": "in",
            "content": {
                "field": "files.access",
                "value": ["open"],
            },
        },
        {
            "op": "in",
            "content": {
                "field": "cases.samples.tissue_type",
                "value": ["tumor"],
            },
        },
        {
            "op": "in",
            "content": {
                "field": "cases.samples.sample_type",
                "value": ["primary tumor"],
            },
        },
    ],
}

params = {
    "filters": json.dumps(filters),
    "fields": ",".join(
        [
            "file_id",
            "file_name",
            "cases.submitter_id",
            "cases.samples.submitter_id",
            "cases.samples.portions.submitter_id",
            "cases.samples.portions.analytes.submitter_id",
            "cases.samples.portions.analytes.aliquots.submitter_id",
            "cases.project.project_id",
        ]
    ),
    "format": "JSON",
    "size": max_size,
}

# files_endpt: https://docs.gdc.cancer.gov/API/Users_Guide/Getting_Started/#query-files

response = requests.get(files_endpt, params=params)

# write metadata
data = []
for x in json.loads(response.content.decode("utf-8"))["data"]["hits"]:
    case = x["cases"][0]
    sample = case["samples"][0]
    portion = sample["portions"][0]
    analyte = portion["analytes"][0]
    aliquot = analyte["aliquots"][0]

    data.append(
        {
            "file_id": x["file_id"],
            "file_name": x["file_name"],
            "case_id": case["submitter_id"],
            "sample_id": sample["submitter_id"],
            "portion_id": portion["submitter_id"],
            "analyte_id": analyte["submitter_id"],
            "aliquots_id": aliquot["submitter_id"],
            "project_id": case["project"]["project_id"],
        }
    )
df = pd.DataFrame(data)
assert len(df) < max_size
assert df["file_name"].str.endswith(suffix).all()

df.to_csv(data_folder + "metadata.csv", index=False)

print(
    f"Average number of sequencing files per case:",
    f"{df.groupby('case_id').size().mean():0.3f}",
)



# data_endpt: https://docs.gdc.cancer.gov/API/Users_Guide/Getting_Started/#download-data
# get the rna-seq file data for each file_id (10 at a time)

for i in trange(0, len(df), batch_size):

    # retrieve 10 files at a time (batch_size=10)
    file_ids = df["file_id"].iloc[i : i + batch_size].to_list() # 10 file_ids
    params = {"ids": file_ids} # for querying

    response = requests.post(
        data_endpt,
        data=json.dumps(params), # filter 10 file_ids
        headers={"Content-Type": "application/json"},
    )


    if len(file_ids) == 1: # if only one file, download directly

        # check response.headers and write response.content
        response_head_cd = response.headers["Content-Disposition"]
        file_name = re.findall("filename=(.+)", response_head_cd)[0]
        if file_name.endswith(suffix):
            with open(data_folder + file_name, "wb") as output_file:
                output_file.write(response.content)

    # most likely:
    else: # if more than 1 file, download as tar.gz and extract

        # write to tar.gz because it's multiple files at one time
        # download as tar.gz, write the response.content to tarball folder temp.tar.gz
        with open(data_folder + "temp.tar.gz", "wb") as output_file:
            output_file.write(response.content)

        # read it and extract it to data/temp
        with tarfile.open(data_folder + "temp.tar.gz", "r") as tar:
            tar.extractall(data_folder + "temp")

        # among the files in the temp tar ball
        for root, dirs, files in os.walk(data_folder + "temp"):
            root = Path(root)
            for f in files:
                if f.endswith(suffix):
                    # among the files in the temp tar ball, 
                    # if file ends with suffix, move it to data folder
                    shutil.move(root / f, Path(data_folder) / f) 


        shutil.rmtree(data_folder + "temp")
        os.remove(data_folder + "temp.tar.gz")
