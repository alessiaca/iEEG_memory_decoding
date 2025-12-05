# Across patient decoding using the optimal channel based on the functional connectivity

import numpy as np
import glob
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import nibabel as nib
from scipy.stats import spearmanr, pearsonr, rankdata
from sklearn.metrics import accuracy_score
import sqlite3
from itertools import product
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import py_neuromodulation as nm
import mne
mne.set_log_level('ERROR')
import sys


def decode_WM_load_cross_pat(subject, seconds=5, components=3, sampling_hz=1):
    """Given a set of parameters, compute the similarity of each channel and patient to the r-map calculated on the remaining patients"""

    # Load the db containing the best channels
    path = f"best_channels/{subject}_{seconds}_{components}_{sampling_hz}.db"
    with sqlite3.connect(path) as conn:
        df = pd.read_sql("SELECT * FROM selected_data", conn)

    # Compute the features for each channel 
    X_train = []
    y_train = []
    for _, row in df.iterrows():
        subject_tmp = int(row["subject_tmp"])
        channel = int(row["channel"])
        
        df_label = pd.read_csv(f'WM_data/Subject_0{subject_tmp}.csv')
        y = df_label["Set size"].to_numpy()

        # Select the first 28 WM load trials with 4 and 8 (minimum number across cohort)
        use_idx_4 = np.where(y == 4)[0][:28]
        use_idx_8 = np.where(y == 8)[0][:28]
        use_idx = np.concatenate((use_idx_4, use_idx_8))
        y = y[use_idx]
        if subject_tmp != subject:
            y_train.append(y)
        else:
            y_test = y

        # Load the epoched data from the given subject/channel combination
        epochs = mne.read_epochs(f"WM_data/Subject_0{subject_tmp}.fif")
        epochs.pick(f"CH{channel-1}")
        data = epochs[:150].get_data(tmin=0, tmax=6)
        n_epochs, n_channels, n_samples = data.shape
        ch_names = epochs.info["ch_names"]
        ch_types = epochs.get_channel_types()

        # Concatenate the epochs together to use the feature extraction from py_neuromodulation
        data_flat = data.reshape(n_channels, n_epochs * n_samples)

        # Prepare py_neuromodulation feature extraction
        channels = nm.utils.set_channels(
            ch_names=ch_names,
            ch_types=ch_types,
            used_types=ch_types,
            reference=None,
            target_keywords=None,
        )
        settings = nm.NMSettings.get_fast_compute()
        settings.features.fft = True
        settings.features.welch = False
        settings.features.return_raw = False
        settings.features.raw_hjorth = True
        settings.features.sharpwave_analysis = True
        settings.segment_length_features_ms = 1000
        settings.sampling_rate_features_hz = sampling_hz
        settings.postprocessing.feature_normalization = False
        stream = nm.Stream(
            settings=settings,
            channels=channels,
            verbose=False,
            sfreq=2000,
            line_noise=50
        )

        # Compute features (returns a dataframe)
        features = stream.run(data=data_flat, out_dir="WM_tmp_out", experiment_name=f"{subject}_{channel}", save_csv=False)

        # Get all features from the given channel (apart from the time)
        columns = [column for column in features.columns if f"CH{channel-1}" in column]

        # Organize concatenated features back into epochs and normalize (1 epoch = 6000 milliseconds) 
        features_epochs = []
        for i in range(0, int(np.max(features.time) / 6000)):
            # Get the features from a specific epoch
            features_tmp = features[(features.time <= 6000*(i+1)+1) & (features.time > 6000*i+1000)][columns].to_numpy()

            # Normalize by calculating percent in respect to first entry
            features_tmp_norm = (features_tmp - features_tmp[0, :]) / features_tmp[0, :]

            # Keep only the seconds of interest
            features_tmp_norm = features_tmp_norm[int(-seconds * sampling_hz):, :]

            features_epochs.append(features_tmp_norm)
        features_epochs = np.array(features_epochs)

        # Reshape the features such that features from one epoch (but different time points) are in one row
        X = np.reshape(features_epochs, (features_epochs.shape[0], features_epochs.shape[1] * features_epochs.shape[2]))

        # Select the first 28 trials of each class
        X = X[use_idx]
        if subject_tmp != subject:
            X_train.append(X)
        else:
            X_test = X

    # Train the model using the aggregated data from all other subjects
    model = LogisticRegression()
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=6)),
        ('clf', model)
    ])

    # Fit the full model on all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    pipeline.fit(X_train, y_train)

    # Get training performance
    y_pred = pipeline.predict(X_train)   
    acc = accuracy_score(y_train, y_pred)
    print(f"Subject {subject} Training accuracy {acc}")

    # Predict on data from the current subject
    y_test = shuffle(y_test)
    y_pred = pipeline.predict(X_test)   
    acc = accuracy_score(y_test, y_pred)
    print(f"Subject {subject} Test accuracy {acc}")


if __name__ == "__main__":

    cluster = False
    subject_mode = np.array([2, 3, 6, 7, 8, 9])
    combinations = subject_mode

    # Run
    if cluster:
        input1 = int(sys.argv[1])-1 
        decode_WM_load_cross_pat(combinations[input1])
    else:
        Parallel(n_jobs=1)(delayed(decode_WM_load_cross_pat)(subject) for subject in subject_mode)
