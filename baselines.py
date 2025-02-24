"""
Baseline CoxPH models or unimodal Cox models
1) clinical data, linear CoxPH
2) gene expression data, PCA down to 64 components, linear CoxPH
3) WSI images, average pooling for each patches, PCA, linear CoxPH
"""

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt

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
    clinical_path = config["clinical"]["cleaned_clinical"]
    clinical = pd.read_csv(clinical_path)

    # choose 3 decided features
    clinical = clinical[["gender", "age_diagnosis", "ajcc_stage", "event", "time"]]

    # split the data
    train = clinical.sample(frac=0.8, random_state=123)
    test = clinical.drop(train.index)

    # fit the model, no need for regularization because no.features are small
    clinical_coxph = CoxPHFitter()
    clinical_coxph.fit(train, duration_col="time", event_col="event")
    print("Clinical baseline model coefficients:")
    clinical_coxph.print_summary()

    # evaluate the model on test data
    pred_survival = -clinical_coxph.predict_partial_hazard(test)
    c_index = concordance_index(test['time'], pred_survival, test['event'])
    print("Concordance index for clinical baseline model on test data:", c_index)

    # plot the Kaplan-Meier curve for the predicted hazard scores based on median risk indices
    fig, ax = plt.subplots(figsize=(10,8))
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    high_risk_idx = pred_survival > np.median(pred_survival)
    low_risk_idx = pred_survival <= np.median(pred_survival)
    kmf_high.fit(test['time'][high_risk_idx], test['event'][high_risk_idx], label="High risk")
    kmf_low.fit(test['time'][low_risk_idx], test['event'][low_risk_idx], label="Low risk")
    kmf_high.plot(ax=ax, ci_show=True, show_censors=True)
    kmf_low.plot(ax=ax, ci_show=True, show_censors=True)

    ax.set_title("Kaplan-Meier curve for clinical baseline model")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    plt.legend()
    plt.savefig("evaluation-results/clinical_baseline.png")

    if os.path.isfile("evaluation-results/clinical_baseline.png"):
        print("Saved KM curve plot")
        print("================================================\n")

    return c_index #, log_rank # NOTE



##############################################
#### Baseline gene expression CoxPH model ####
##############################################

def baseline_rna_seq():

    pass



##############################################
#### Baseline WSI CoxPH model ################
##############################################

def baseline_wsi():
    
    pass







if __name__ == "__main__":
    
    c_index_baseline_clinical = baseline_clinical()


    c_index_results = pd.DataFrame(
        {
            "c-index_baseline_clinical": c_index_baseline_clinical,
            # TODO
        }
    )

    df.to_csv("evaluation-results/c-index_results.csv", index=False)
