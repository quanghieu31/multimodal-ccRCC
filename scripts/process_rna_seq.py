import configparser
config = configparser.ConfigParser()
config.read("../config.ini")

from pathlib import Path
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

np.random.seed(0)
random.seed(0)

RNA_DATA_PATH = Path(config["rna"]["rna_path"])

# CPM + log-transform

def process_one_case(case_id, case_files, data_dir=RNA_DATA_PATH):
    """
    read files/samples for a case and
    aggregates data for that case
    """
    # initialize the dataframe
    selected_cols = ["gene_id", "log_cpm_unstranded"]
    concat_data = pd.DataFrame()
    
    # process each tsv_file for a case
    for tsv_file in case_files:
        data = pd.read_csv(data_dir / tsv_file, sep='\t', header=1)
        data = data[data["gene_type"] == "protein_coding"][["gene_id", "unstranded"]]
        # calculate CPM
        data["cpm_unstranded"] = (data["unstranded"] / data["unstranded"].sum()) * 1e6
        # calculate log-transform on CPM     
        data["log_cpm_unstranded"] = np.log1p(data["cpm_unstranded"])
        # select only gene_id and log_cpm_unstranded columns
        data = data[["gene_id", "log_cpm_unstranded"]]
        concat_data = pd.concat([concat_data, data], ignore_index=True)

    # aggregate the samples and transpose (prep for patient-gene matrix later)
    aggregated_data = concat_data.groupby("gene_id").mean().reset_index()
    aggregated_data = aggregated_data.rename(columns={selected_cols[1]: case_id})

    return aggregated_data



def process_all_cases_parallel(cases_files, max_workers=8):
    """
    outputs a case-gene matrix (like above, but in parallel)
    """
    gene_case_matrix = pd.DataFrame(columns=["gene_id"])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {executor.submit(process_one_case, case_id, case_files): case_id for case_id, case_files in cases_files.items()}
        
        for future in tqdm(as_completed(future_to_case), total=len(future_to_case)):
            case_id = future_to_case[future]
            case_data = future.result()
            if case_data is not None:
                gene_case_matrix = gene_case_matrix.merge(case_data, on="gene_id", how="outer")

    # transpose the matrix
    case_gene_matrix = gene_case_matrix.set_index("gene_id").T.reset_index().rename_axis(None, axis=1)
    case_gene_matrix.columns.values[0] = "submitter_id"
    return case_gene_matrix




if __name__ == "__main__":

    # build a dictionary of {case_id: list_of_tsv_rna_file_paths}
    metadata = pd.read_csv(config["rna"]["rna_metadata"])
    cases = metadata["case_id"].unique()
    cases_files = {case: metadata[metadata["case_id"] == case]["file_name"].values for case in cases}

    case_gene_matrix = process_all_cases_parallel(cases_files)
    case_gene_matrix.to_csv(config["rna"]["cleaned_rna"], index=False)

    print("cleaned_rna.csv exists?", os.path.isfile(config["rna"]["cleaned_rna"]))