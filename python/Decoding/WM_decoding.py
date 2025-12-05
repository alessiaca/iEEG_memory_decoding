# Run decoding on BIH server with parallel processing
# Test multiple parameters in grid search 

import py_neuromodulation as nm
import numpy as np
import mne
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold, RepeatedStratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import sqlite3
from sklearn.svm import SVC
import sys
from joblib import Parallel, delayed
from itertools import product
import os
    
def run_cross_val(X, y, groups, n_splits=8, repetitions=20):
    """Returns average test-score from repeated stratified crossvalidation"""

    model = SVC(kernel='linear')
    pipeline = Pipeline([
    ('scaler', RobustScaler()),      
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
    
    # Load the label
    df = pd.read_csv(f'../../../Data_processed/Re_referenced/Data_Subject_{subject}.csv')
    y = df["SetSize"].to_numpy()

    # Select the first 28 WM load trials with 4 and 8 (minimum number across cohort)
    use_idx_4 = np.where(y == 4)[0][:24]
    use_idx_8 = np.where(y == 8)[0][:24]
    use_idx = np.concatenate((use_idx_4,use_idx_8))
    y = y[use_idx]

    # Load the epoched data from the given subject/channel combination
    epochs = mne.read_epochs(f"../../../Data_processed/Re_referenced/Data_Subject_{subject}.fif")
    ch_names = epochs.ch_names
    for ch in ch_names:
        data = epochs.copy().pick(ch).get_data()
        print(np.sum(np.isnan(data)))

    sfreq = epochs.info["sfreq"]
    epochs.pick(ch_names[channel])
    data = epochs[use_idx].get_data(tmin=0, tmax=6)
    n_epochs, n_channels, n_samples = data.shape
    ch_names = epochs.info["ch_names"]
    ch_types = epochs.get_channel_types()

    # Concatenate the epochs together to use the feature extraction from py_neuromodulation
    data_flat = data.reshape(n_channels, n_epochs * n_samples)
    
    decoding_accuracy_all = []

    # Loop over the grid search parameters (oversampling parameters)
    for sampling_hz in [1, 3, 5]:

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
        settings.preprocessing = settings.preprocessing[:-1]
        stream = nm.Stream(
            settings=settings,
            channels=channels,
            verbose=False,
            sfreq=sfreq,
            line_noise=50
        )

        # Compute features (returns a dataframe)
        features = stream.run(data=data_flat, out_dir="WM_tmp_out", experiment_name=f"{subject}_{channel}", save_csv=False)

        # Get all features from the given channel (apart from the time)
        columns = [column for column in features.columns if ch_names[channel] in column]

        # Organize concatenated features back into epochs and normalize (1 epoch = 6000 milliseconds) 
        features_epochs = []
        for i in range(0, int(np.max(features.time) / 6000)):
            # Get the features from a specific epoch
            features_tmp = features[(features.time <= 6000*(i+1)+1) & (features.time > 6000*i+1000)][columns].to_numpy()

            # Normalize by calculating percent in respect to first entry
            features_tmp_norm = (features_tmp - features_tmp[0, :]) / features_tmp[0, :]

            # Keep only the seconds of interest
            features_tmp_norm = features_tmp_norm[(-2*sampling_hz)-1:, :]

            features_epochs.append(features_tmp_norm)
        features_epochs = np.array(features_epochs)
        
        # Reshape features matrix and upsample
        X = np.reshape(features_epochs, (features_epochs.shape[0] * features_epochs.shape[1], features_epochs.shape[2]))

        # Upsample the labels to match the features
        upsample_factor = int(sampling_hz*2)+1
        y_upsample = np.repeat(y, upsample_factor)
        
        # Define groups to avoid data leakage between training and test set
        groups = np.repeat(np.arange(n_epochs), upsample_factor)

        row = {'subject': subject,
                'channel': channel+1,
                'sampling_hz': sampling_hz,
                'perm': -1,
                'shuffled': 0,
                'accuracy': run_cross_val(X, y_upsample, groups)
            }
        decoding_accuracy_all.append(row)
        n_perm = 500
        for k in range(n_perm):
            y_perm = np.random.permutation(y)
            y_perm_upsample = np.repeat(y_perm, upsample_factor)
            row = {
                'subject': subject,
                'channel': channel+1,
                'sampling_hz': sampling_hz,
                'perm': k,
                'shuffled': 1,
                'accuracy': run_cross_val(X, y_perm_upsample, groups)
            }
            decoding_accuracy_all.append(row)

    df = pd.DataFrame(decoding_accuracy_all)
    df.to_csv(f"Results/WM_load_decoding_performance_{subject}_{channel+1}.csv", index=False)


if __name__ == "__main__":

    cluster = False

    # Grid search parameters to parallelize over
    subjects = np.arange(13, 16)
    channels = np.arange(64)
    combinations = list(product(subjects, channels))

    # Run
    if cluster:
        input1 = int(sys.argv[1])-1 
        decode_WM_load(combinations[input1][0], combinations[input1][1])
    else:
        Parallel(n_jobs=1)(delayed(decode_WM_load)(subject, channel) for subject, channel in combinations)