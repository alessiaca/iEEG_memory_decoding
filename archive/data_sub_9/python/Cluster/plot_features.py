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
from scipy.stats import wilcoxon, zscore
import sqlite3
import sys
import seaborn as sns
from joblib import Parallel, delayed
from itertools import product
from scipy.signal import butter, filtfilt, hilbert, welch
import os
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.svm import SVC
import matplotlib
matplotlib.use("TkAgg")


def plot_features(shift_ms=100, length_ms=800):

    fig, axes = plt.subplots(3,3,figsize=(10,10))
    for subject, ax in zip(range(1, 10), axes.flatten()):

        # Load the features
        features_sub = np.load(f"features_bip/Subject_{subject}_bip_Hz_5_length_800.npy")
        
        # Normalize using z-score
        baseline = features_sub[:, :, :3, :].mean(axis=-2)[:, :, np.newaxis, :]
        features_sub = (features_sub[:, :, :, :] - baseline) / baseline

        # Clip features
        #features_sub = np.clip(features_sub, -100, 100)

        for channel in range(features_sub.shape[1]):

            # Select the channel of interest
            features_sub_chan = features_sub[:, :, :, :] #- features_sub[:28, :, :, :]

            ax.imshow(features_sub_chan.mean(axis=(0, 1)).T, aspect='auto', cmap='viridis')
            #plt.colorbar()
            ax.set_title(f"Subject {subject}")
            """ax.axvline(10)
            ax.axvline(30)
            ax.axvline(60)"""

        # Plot feature correlation
        """reshaped = features_sub.reshape(-1, 8)

        # Compute correlation matrix (8x8)
        corr_matrix = np.corrcoef(reshaped, rowvar=False)

        # Optional: Create labels for features (or use actual ones if you have them)
        labels = [f'Feature {i+1}' for i in range(8)]

        # Plot using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, xticklabels=labels, yticklabels=labels,
                    cmap='coolwarm', center=0, square=True, fmt=".2f")
        plt.title('Correlation Matrix (8 Features)')
        plt.tight_layout()"""
    plt.show()


if __name__ == "__main__":

    plot_features()