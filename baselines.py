"""
Baseline CoxPH models or unimodal Cox models
1) clinical data, linear CoxPH
2) gene expression data, PCA down to 64 components, linear CoxPH
3) WSI images, average pooling for each patches, PCA, linear CoxPH
"""

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter, KaplanMeierFitter
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from utils import display_km_curves

import configparser
config = configparser.ConfigParser()
config.read("config.ini")

random.seed(0)
np.random.seed(0)

##############################################
#### Baseline clinical CoxPH model ###########
##############################################

def baseline_clinical():
    print("CLINICAL BASELINE...")
    clinical = pd.read_csv(config["clinical"]["cleaned_clinical_json"])

    # drop submitter_id
    clinical = clinical.drop(["submitter_id"], axis=1)

    # split the data
    train = clinical.sample(frac=0.8, random_state=123)
    test = clinical.drop(train.index)

    # fit the model, no need for regularization because no.features are small
    clinical_coxph = CoxPHFitter(penalizer=0.01)
    # why need penalizer here? apply l2 regularization to help in cases of high collinearity due to sparsity in the one-hot encoding features
    # READ: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model 
    clinical_coxph.fit(train, duration_col="time", event_col="event")
    print("Clinical baseline model coefficients:")
    clinical_coxph.print_summary()

    # evaluate the model on test data
    pred_risk = clinical_coxph.predict_partial_hazard(test.drop(["time", "event"], axis=1))
    c_index = concordance_index(test['time'], -pred_risk, test['event'])
    print("c-index for clinical model on test data:", c_index)

    # plot the Kaplan-Meier curve for the predicted hazard scores based on median risk indices
    fig, ax = plt.subplots(figsize=(10,8))
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    high_risk_idx = pred_risk > np.median(pred_risk)
    low_risk_idx = pred_risk <= np.median(pred_risk)
    kmf_high.fit(test['time'][high_risk_idx], test['event'][high_risk_idx], label="High risk")
    kmf_low.fit(test['time'][low_risk_idx], test['event'][low_risk_idx], label="Low risk")
    kmf_high.plot(ax=ax, ci_show=True, show_censors=True)
    kmf_low.plot(ax=ax, ci_show=True, show_censors=True)

    ax.set_title("Kaplan-Meier curve for clinical baseline model")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    plt.legend()
    plt.savefig("evaluation-results/clinical-baseline.png")
    print()

    return c_index



##############################################
#### Baseline gene expression CoxPH model ####
##############################################

def baseline_rna_seq():
    """
    PCA on ~20000 genes to get 64 principal components
        after considering the impacts of different n_components for the PCA, 64 is the best number (highest c-index among the validations)
    Then fit them to a CoxPH
    """
    print("RNA-SEQ BASELINE...")
    
    # collect time_event data and rna-seq data
    clinical = pd.read_csv(config["clinical"]["cleaned_clinical_json"])
    time_event = clinical[["time", "event", "submitter_id"]]
    df_rna = pd.read_csv(config["rna"]["cleaned_rna"])

    n_components = 16   # TODO after a few experiments, but why really

    # fit pca on the columns not including the "submitter_id" column
    pca = PCA(n_components=n_components, random_state=0) 
    features_pca = pca.fit_transform(df_rna.iloc[:, 1:].values)
    print("explained variance:", pca.explained_variance_ratio_)
    # then convert to dataframe to merge with time and event (labels)
    features_pca_df = pd.DataFrame(features_pca, columns=[f"pc{i}" for i in range(1, n_components + 1)])
    # add the submitter_id back to pca 
    features_pca_df["submitter_id"] = df_rna["submitter_id"].values
    final_pca_data = pd.merge(features_pca_df, time_event, on="submitter_id").drop("submitter_id", axis=1)

    # split
    train, test = train_test_split(final_pca_data, test_size=0.3, random_state=0)
    # fit
    coxph = CoxPHFitter()
    coxph.fit(train, duration_col="time", event_col="event")
    # train c-index
    print(f"train c-index:", coxph.concordance_index_)

    pred_risk = coxph.predict_partial_hazard(test.drop(["time", "event"], axis=1))
    test_c_index = concordance_index(test['time'], -pred_risk, test['event'])
    print("test c-index:", test_c_index)
    print()

    display_km_curves(test, pred_risk, "RNA-seq", save_figure=True)

    return test_c_index



##############################################
#### Baseline WSI CoxPH model ################
##############################################

def baseline_wsi():
    # TODO
    pass




if __name__ == "__main__":
    
    c_index_baseline_clinical = baseline_clinical()
    c_index_baseline_rna_seq = baseline_rna_seq()

    c_index_results = pd.DataFrame({
            "model": ["baseline_clinical", "baseline_rna-seq"],
            "c-index": [c_index_baseline_clinical, c_index_baseline_rna_seq]
        }
    )

    c_index_results.to_csv("evaluation-results/c-index-results.csv", index=False)
