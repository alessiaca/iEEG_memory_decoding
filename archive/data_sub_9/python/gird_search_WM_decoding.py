# Run decoding on BIH server with parallel processing

import py_neuromodulation as nm
import numpy as np
import os
import mne
from catboost import CatBoostClassifier
from sklearn import metrics, model_selection, linear_model, svm
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import matplotlib
import matplotlib.pyplot as plt
import sqlite3
import sys
from scipy.stats import zscore
from joblib import Parallel, delayed
from itertools import product
import random
seed = 42

def run_cross_val(X, y, n_folds=6, n_components=4):

    model = LogisticRegression()

    pipeline = Pipeline([
    ('scaler', StandardScaler()),      
    ('pca', PCA(n_components=n_components)),     
    ('clf', model)     
])  
    # Use cross-validation (e.g. stratified K-fold for classification)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True)

    # Run cross-validation
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    return scores.mean()


def decode_WM(combination, referencing, n_runs):

    seconds, components, sampling_hz = combination
    subjects = np.arange(1, 10)
    for subject in subjects:

        # Load the label
        #df = pd.read_csv(f'Data_Subject_0{subject}.csv')
        df = pd.read_csv(f'../../Processed_data/merged/Data_Subject_0{subject}.csv')

        # Set the label
        y = df["Set size"].to_numpy()
        use_idx_4 = np.where(y == 4)[0][:28]
        use_idx_8 = np.where(y == 8)[0][:28]
        use_idx = np.concatenate((use_idx_4, use_idx_8))
        y = y[use_idx]

        # Extract the features
        #epochs = mne.read_epochs(f"merged/Data_Subject_0{subject}.fif")
        epochs = mne.read_epochs(f"../../Processed_data/merged/Data_Subject_0{subject}.fif")
        channel = 0
        epochs.pick(f"CH{channel}")

        ch_names = epochs.info["ch_names"]
        sfreq = epochs.info["sfreq"]
        ch_types = epochs.get_channel_types()

        channels = nm.utils.set_channels(
            ch_names=ch_names,
            ch_types=ch_types,
            used_types=ch_types,
            reference=None,
            target_keywords=None,
        )

        # Get epochs data
        data = epochs[:110].get_data(tmin=0, tmax=6)
        n_epochs, n_channels, n_samples = data.shape
        data_flat = data.reshape(n_channels, n_epochs * n_samples)

        # PN settings
        settings = nm.NMSettings.get_fast_compute()
        settings.features.fft = True
        settings.features.welch = False
        settings.features.return_raw = False
        settings.features.raw_hjorth = True
        settings.features.sharpwave_analysis = True
        settings.segment_length_features_ms = 1000
        settings.sampling_rate_features_hz = sampling_hz
        settings.postprocessing.feature_normalization = False

        # Compute features
        stream = nm.Stream(
            settings=settings,
            channels=channels,
            verbose=False,
            sfreq=2000,
            line_noise=50
        )

        features = stream.run(data=data_flat, out_dir="", experiment_name=f"", save_csv=False)

        for ch in ch_names:

            # Get all channel
            columns = [column for column in features.columns if ch in column]

            if len(columns) > 0:

                # Reshape into epochs
                features_long = []
                for i in range(0, int(np.max(features.time) / 6000)):
                    features_epoch = features[(features.time <= 6000*(i+1)+1) & (features.time > 6000*i+1000)]
                    # Select only the columns of interest
                    features_epoch = features_epoch[columns].to_numpy()
                    # Normalize by calculating percent in respect to first entry, keep only the last 2 seconds
                    features_epoch_norm = (features_epoch[int(-seconds * sampling_hz):, :] - features_epoch[0, :]) / features_epoch[0, :]
                    features_long.append(features_epoch_norm)

                # Normalize
                features_long = np.array(features_long)

                # Reshape the features by concatenating the features from one epoch (but different time points) in one row
                X = np.reshape(features_long, (features_long.shape[0], features_long.shape[1] * features_long.shape[2]))
                
                # Choose only the first 2 sessions (100 samples)
                X = X[use_idx]

                # Compute performance of 100 runs (difference due to randomness in umsampling)
                results = []  # list to store rows
                for run in range(n_runs):
                    # Run 10-fold cross-validation
                    mean_accuracy = run_cross_val(X, y, n_components=components)
                
                    # Collect results in a list
                    results.append([subject, ch, mean_accuracy, run])
                
                # Create a DataFrame from the collected results
                df = pd.DataFrame(results, columns=['subject', 'channel', 'accuracy', 'run'])
                
                # Connect to SQLite database once
                #conn = sqlite3.connect(f"decoding_results/{referencing}_n_runs_{n_runs}.db")
                conn = sqlite3.connect(f"{referencing}_n_runs_{n_runs}.db")
                
                # Append the entire dataframe to the database
                df.to_sql("selected_data", conn, if_exists="append", index=False)
                
                # Commit and close the connection
                conn.commit()
                conn.close()


if __name__ == "__main__":

    # Set parameters
    referencing = "none"
    n_runs = 200

    # Options to loop over
    seconds = [2, 1.5, 1]
    components = [3, 4, 5, 6]
    sampling_hz = [ 2, 5, 10]

    # Prepare job list
    combinations = list(product(seconds, components, sampling_hz))

    # Run in parallel
    #input1 = int(sys.argv[1]) -1 
    Parallel(n_jobs=1)(delayed(decode_WM)(comb, referencing, n_runs) for comb in combinations)
    #decode_WM(input1, combinations, referencing, n_runs)