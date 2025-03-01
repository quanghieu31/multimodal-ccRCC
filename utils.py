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


def display_km_curves_fusion(risks, times, events, title_name, save_figure=False):
    risks = np.array(risks)
    times = np.array(times)
    events = np.array(events)

    fig, ax = plt.subplots(figsize=(10, 8)) 
    
    high_risk_idx = risks > np.median(risks)
    low_risk_idx = risks <= np.median(risks)
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    # fit low risk
    kmf_low.fit(times[low_risk_idx], event_observed=events[low_risk_idx], label='Low risk')
    kmf_low.plot_survival_function(ax=ax, ci_show=True)

    # fit high risk
    kmf_high.fit(times[high_risk_idx], event_observed=events[high_risk_idx], label='High risk')
    kmf_high.plot_survival_function(ax=ax, ci_show=True)
    ax.set_title(f"Kaplan-Meier curve for final model on {title_name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    plt.legend()
    if save_figure:
        plt.savefig(f"evaluation-results/{title_name}-baseline.png")
    else:
        plt.show()


def concordance_index(hazards, times, events):
    """
    computes the c-index for survival prediction
    - hazards: predicted risk scores (higher means higher risk)
    - times: observed survival times
    - events: event indicators (1 if event occurred, 0 if censored)
    """
    n = len(times)
    concordant = 0.0
    permissible = 0.0
    for i in range(n):
        for j in range(n):
            # only compare if i had an event and its time is earlier than j
            if times[i] < times[j] and events[i] == 1:
                permissible += 1
                if hazards[i] > hazards[j]:
                    concordant += 1
                elif hazards[i] == hazards[j]:
                    concordant += 0.5
    return concordant / permissible if permissible > 0 else 0