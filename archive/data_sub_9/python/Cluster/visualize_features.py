# Visualize features during 4 vs 8 WM load

import py_neuromodulation as nm
import numpy as np
import mne
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from scipy.stats import wilcoxon
import sqlite3
import sys
from joblib import Parallel, delayed
from itertools import product
from scipy.signal import welch
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# Limit to 10 features, compute per each time point, use 1 sec for estimation, compute performance for each 8 sec (fixation as control), play raound with normalization
# Logistic regression with regularization, visualize one line per patient with sem across channels, for different measures
# Compute features: store as npy
# Compute decoding performance: store as db
# Visualize

def compute_hjorth_features(array):
    # Compute the Hjorth parameters
    activity = np.var(array)
    mobility = np.std(np.diff(array)) / np.std(array)
    complexity = np.std(np.diff(np.diff(array))) / np.std(np.diff(array))
    return activity, mobility, complexity

def compute_welch_features(array):
    # Compute power spectral density with Welch
    frequencies, psd = welch(array, fs=2000, nperseg=2000)  # nperseg = window length in samples

    # Define band masks
    theta_mask = (frequencies >= 4) & (frequencies <= 7)
    alpha_mask = (frequencies >= 8) & (frequencies <= 12)
    beta_mask  = (frequencies >= 13) & (frequencies <= 35)
    gamma_mask = (frequencies >= 40) & (frequencies <= 100)

    # Integrate PSD over each band
    df = frequencies[1] - frequencies[0]
    theta_power = np.sum(psd[theta_mask]) * df
    alpha_power = np.sum(psd[alpha_mask]) * df
    beta_power  = np.sum(psd[beta_mask])  * df
    gamma_power = np.sum(psd[gamma_mask]) * df

    return theta_power, alpha_power, beta_power, gamma_power


def test():
    
    features_4 = []
    features_8 = []
    for subject in range(1, 10):
        
        # Load the label
        df = pd.read_csv(f'WM_data/Subject_0{subject}.csv')
        y = df["Set size"].to_numpy()

        # Select the first 28 WM load trials with 4 and 8 (minimum number across cohort)
        use_idx_4 = np.where(y == 4)[0][:28]
        use_idx_8 = np.where(y == 8)[0][:28]
        use_idx = np.concatenate((use_idx_4, use_idx_8))
        y = y[use_idx]

        # Load the epoched data from the given subject/channel combination
        tmin = 0
        tmax = 8
        epochs = mne.read_epochs(f"WM_data/Subject_0{subject}.fif")
        data = epochs[use_idx].get_data(tmin=tmin, tmax=tmax)

        # Reshape into smaller segments
        data_reshaped = data.reshape(data.shape[0], data.shape[1], tmax-tmin, 2000)

        # Compute hjorth features
        #hjorth_features = np.apply_along_axis(compute_hjorth_features, axis=-1, arr=data_reshaped)

        # Normalize to baseline
        #baseline = hjorth_features[:, :, 0, :][:, :, np.newaxis, :]
        #hjorth_features = (hjorth_features - baseline) /baseline

        # Compute welch features
        welch_features = np.apply_along_axis(compute_welch_features, axis=-1, arr=data_reshaped)
        baseline = welch_features[:, :, 0, :][:, :, np.newaxis, :]
        welch_features = (welch_features - baseline) /baseline
        features_4.append(welch_features[28:, :, 1:, :])
        features_8.append(welch_features[:28, :, 1:, :])


    # Compare statistically
    for i in range(4):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for j in range(7):
            features_4_tmp = np.array([y for x in features_4 for y in x[:, :, j, i].flatten()])
            features_8_tmp = np.array([y for x in features_8 for y in x[:, :, j, i].flatten()])
            r, p = wilcoxon(features_4_tmp, features_8_tmp)

            ax = axes[j]
            ax.boxplot([features_4_tmp, features_8_tmp], labels=["4", "8"], showfliers=False)
            ax.set_title(f"Segment {j+1}\n(p = {p:.1e})")
            ax.set_ylabel("Feature")
            ax.grid(True)

        fig.suptitle(f"Hjorth Feature {i} Comparison Across 8 Time Segments", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":

    test()