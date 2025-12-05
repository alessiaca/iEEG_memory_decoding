import py_neuromodulation as nm
import numpy as np
import mne
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from scipy.stats import wilcoxon, zscore
import sqlite3
import sys
from joblib import Parallel, delayed
from itertools import product
from scipy.signal import butter, filtfilt, hilbert, welch
import os
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
import matplotlib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
matplotlib.use("TkAgg")


def run_cross_val(X):
    """Returns average test-score from 7 folds"""

    n_samples_class = int(X.shape[0]/2)
    y = np.array([0]*n_samples_class + [1]*n_samples_class)
    model = LinearDiscriminantAnalysis()
    """model = SVC(
    kernel='rbf',       # or 'linear', 'poly', 'sigmoid'
    C=1.0,              # Regularization (smaller = more regularization)
    gamma='scale'       # Controls kernel complexity
)
    model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0)"""
    pipeline = Pipeline([     
    ('clf', model)])  
    cv = StratifiedKFold(n_splits=2, shuffle=True)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    cv_results = cross_validate(
        pipeline, X, y, 
        cv=2, 
        return_train_score=True, 
        scoring='accuracy'
        )

    # Print scores
    print("Train scores:", cv_results['train_score'])
    print("Test scores :", cv_results['test_score'])
    return scores.mean()


def decode_WM_load(subject, channel, shift_ms=100, length_ms=800):

    # Load the features
    features_sub = np.load(f"features/Subject_{subject}_{shift_ms}_{length_ms}.npy")
    
    # Normalize using z-score
    features_sub = zscore(features_sub, axis=-2)

    # Clip features
    features_sub = np.clip(features_sub, -2.5, 2.5)

    # Select the channel of interest
    features_sub_chan = features_sub[:, channel, :, :]

    n_runs = 20
    n_perm = 200
    decoding_accuracy_all = []

    for i in range(features_sub_chan.shape[1]):
        row = {
                'subject': subject,
                'channel': channel+1,
                'period': i,
                'perm': -1,
                'shuffled': 0,
                'accuracy': np.array([run_cross_val(features_sub_chan[:, i, :]) for k in range(n_runs)]).mean()
            }
        decoding_accuracy_all.append(row)
        for k in range(n_perm):
            idx_perm = np.random.permutation(features_sub_chan.shape[0])
            row = {
                'subject': subject,
                'channel': channel+1,
                'period': i,
                'perm': k,
                'shuffled': 1,
                'accuracy': np.array([run_cross_val(features_sub_chan[idx_perm, i, :]) for k in range(n_runs)]).mean()
            }
            decoding_accuracy_all.append(row)

    df = pd.DataFrame(decoding_accuracy_all)
    df.to_csv(f"decoding_results_2/WM_load_decoding_performance_{subject}_{channel+1}_{shift_ms}_{length_ms}.csv", index=False)


if __name__ == "__main__":

    cluster = False

    # Grid search parameters to parallelize over
    subjects = np.arange(1, 10)
    channels = np.arange(64)
    combinations = list(product(subjects, channels))

    # Run
    if cluster:
        input1 = int(sys.argv[1])-1 
        decode_WM_load(combinations[input1][0], combinations[input1][1])
    else:
        Parallel(n_jobs=1)(delayed(decode_WM_load)(subject, channel) for subject, channel in combinations)