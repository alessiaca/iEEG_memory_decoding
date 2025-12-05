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
from scipy.signal import butter, filtfilt, hilbert, welch
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# Limit to 10 features, compute per each time point, use 1 sec for estimation, compute performance for each 8 sec (fixation as control), play raound with normalization
# Logistic regression with regularization, visualize one line per patient with sem across channels, for different measures
# Compute features: store as npy
# Compute decoding performance: store as db
# Visualize

def compute_line_length(array):
    return np.sum(np.abs(np.diff(array)))


def compute_hjorth_features(array):
    # Compute the Hjorth parameters
    activity = np.var(array)
    mobility = np.std(np.diff(array)) / np.std(array)
    complexity = np.std(np.diff(np.diff(array))) / np.std(np.diff(array))
    return activity, mobility, complexity

def compute_welch_features(array):
    # Compute power spectral density with Welch
    nperseg = min(len(array), 1024)
    frequencies, psd = welch(array, fs=2000, nperseg=nperseg)  # nperseg = window length in samples

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


def compute_features(subject, shift_ms=100, length_ms=1000):
        
    # Load the label
    df = pd.read_csv(f'WM_data/Subject_0{subject}.csv')
    y = df["Set size"].to_numpy()

    # Select the first 28 WM load trials with 4 and 8 (minimum number across cohort)
    use_idx_4 = np.where(y == 4)[0][:28]
    use_idx_8 = np.where(y == 8)[0][:28]
    use_idx = np.concatenate((use_idx_4, use_idx_8))
    y = y[use_idx]

    # Load the epoched data from the given subject/channel combination
    epochs = mne.read_epochs(f"WM_data/Subject_0{subject}.fif")
    data = epochs[use_idx].get_data()
    n_samples = data.shape[-1]

    # Transform the ms into samples
    shift_samples = shift_ms * 2
    length_samples = length_ms * 2

    # Reshape into smaller segments
    idx_windows = np.arange(0, n_samples, shift_samples)
    idx_windows = idx_windows[idx_windows + length_samples < n_samples]
    data_reshaped = np.array([data[:, :, idx:idx+length_samples] for idx in idx_windows])
    data_reshaped = np.moveaxis(data_reshaped, 0, -2)

    # Compute hjorth features
    hjorth_features = np.apply_along_axis(compute_hjorth_features, axis=-1, arr=data_reshaped)

    # Compute welch features
    welch_features = np.apply_along_axis(compute_welch_features, axis=-1, arr=data_reshaped)

     # Compute theta-gamma pp pa coupling
    line_length_features = np.apply_along_axis(compute_line_length, axis=-1, arr=data_reshaped)[:, :, :, np.newaxis]

    # Stack features
    features_sub = np.concatenate((hjorth_features, welch_features, line_length_features), axis=-1)

    # Save features
    np.save(f"../../../Features/Subject_{subject}.npy", features_sub)

if __name__ == "__main__":

    cluster = False

    subjects = np.arange(1, 10)

     # Run
    if cluster:
        input1 = int(sys.argv[1])-1
        compute_features(subjects[input1])
    else:
        Parallel(n_jobs=1)(delayed(compute_features)(subject) for subject in subjects)