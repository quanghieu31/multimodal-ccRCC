from sklearn.decomposition import PCA
from lifelines import KaplanMeierFitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def display_km_curves(test_df, pred_risk, title_name, save_figure=False):
    fig, ax = plt.subplots(figsize=(10,8))
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    high_risk_idx = pred_risk > np.median(pred_risk)
    low_risk_idx = pred_risk <= np.median(pred_risk)
    kmf_high.fit(test_df['time'][high_risk_idx], test_df['event'][high_risk_idx], label="High risk")
    kmf_low.fit(test_df['time'][low_risk_idx], test_df['event'][low_risk_idx], label="Low risk")
    kmf_high.plot(ax=ax, ci_show=True, show_censors=True)
    kmf_low.plot(ax=ax, ci_show=True, show_censors=True)

    ax.set_title(f"Kaplan-Meier curve for {title_name} baseline model")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    plt.legend()
    if save_figure:
        plt.savefig(f"evaluation-results/{title_name}-baseline.png")
    else:
        plt.show()
