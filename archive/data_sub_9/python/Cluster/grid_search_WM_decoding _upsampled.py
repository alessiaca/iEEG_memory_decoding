# Run decoding on BIH server with parallel processing
# Test multiple parameters in grid search 

import py_neuromodulation as nm
import numpy as np
import mne
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import sqlite3
import sys
from joblib import Parallel, delayed
from itertools import product
import os
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")


def run_cross_val(X, y, groups, n_splits=5, repetitions=20):
    """Returns average test-score from repeated 5 folds run"""

    model = SVC(kernel='linear')
    pipeline = Pipeline([
    ('scaler', StandardScaler()),      
    ('clf', model)
    ])
    
    scores_all = []
    for _ in range(repetitions):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', groups=groups)
        scores_all.append(scores.mean())

    return np.array(scores_all).mean()


def decode_WM_load(subject, channel):
    """Calculates the WM load decoding accuracy from a given channel using a series of grid search parameters"""

    sampling_hz = 5  # Hz
    
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
    epochs = mne.read_epochs(f"../../../Processed_data/merged/Data_Subject_0{subject}.fif")
    epochs = epochs[use_idx]
    epochs.pick(f"CH{channel}")
    data = epochs.get_data(tmin=0, tmax=6)
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
    columns = [column for column in features.columns if f"CH{channel}" in column]

    # Organize concatenated features back into epochs and normalize (1 epoch = 6000 milliseconds) 
    features_epochs = []
    for i in range(0, int(np.max(features.time) / 6000)):
        # Get the features from a specific epoch
        features_tmp = features[(features.time <= 6000*(i+1)+1) & (features.time > 6000*i+1000)][columns].to_numpy()

        # Normalize by calculating percent in respect to first entry
        features_tmp_norm = (features_tmp - features_tmp[0, :]) / features_tmp[0, :]

        # Keep only the seconds of interest
        features_tmp_norm = features_tmp_norm[-2*sampling_hz:, :]

        features_epochs.append(features_tmp_norm)
    features_epochs = np.array(features_epochs)

    # Reshape the features such that features from one epoch (but different time points) are in one row
    X = np.reshape(features_epochs, (features_epochs.shape[0] * features_epochs.shape[1], features_epochs.shape[2]))
    plt.figure()
    for i in range(15):
        plt.subplot(15,1, i+1)
        plt.hist(X[:, i].flatten())
    plt.show()

    # Upsample the labels to match the features
    upsample_factor = int(sampling_hz*2)
    y = np.repeat(y, upsample_factor)

    groups = np.repeat(np.arange(n_epochs), upsample_factor)

    # Compute performance of 200 runs (difference due to shuffling in k-fold)
    n_runs = 200
    results = [[subject, channel+1, run_cross_val(X, y, groups), run, 0] for run in range(n_runs)]

    # Compute the performance of 200 runs with shuffled labels
    results_shuffled = [[subject, channel+1, run_cross_val(X, shuffle(y, random_state=run), groups), run, 1] for run in range(n_runs)]

    # Create a DataFrame from the collected results
    combined_results = results + results_shuffled
    results_df = pd.DataFrame(combined_results, columns=['subject', 'channel', 'seconds', 'components', 'sampling_hz', 'accuracy', 'run', 'shuffled'])
    
    # Write to database
    with sqlite3.connect(f"decoding_results/{subject}_{channel}.db", timeout=30) as conn:
        results_df.to_sql("selected_data", conn, if_exists="append", index=False)


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