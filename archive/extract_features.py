# Extract features for each subject

import py_neuromodulation as nm
import numpy as np
import pandas as pd
import os
import mne

normalize = True
norm_method = "percent"
referencing = "bip"
n_split = 2  # How many features over time per epoch

subjects = np.arange(1, 10)

for subject in subjects:

    # Load the epochs
    epochs = mne.read_epochs(f"..\data_epochs\\re_referenced\Data_Subject_0{subject}_{referencing}.fif")

    ch_names = epochs.info["ch_names"]
    ch_types = epochs.get_channel_types()

    channels = nm.utils.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        used_types=ch_types,
        reference=None,
        target_keywords=None,
    )

    # Get epochs data
    data = epochs.get_data(tmin=3, tmax=6)
    n_epochs, n_channels, n_samples = data.shape
    data_flat = data.reshape(n_channels, n_epochs * n_samples)

    # PN settings
    settings = nm.NMSettings.get_fast_compute()
    settings.features.fft = True
    settings.features.welch = False
    settings.features.return_raw = False
    settings.features.raw_hjorth = False
    settings.segment_length_features_ms = 1000 #int(3000 / n_split)
    settings.sampling_rate_features_hz = 5
    settings.postprocessing.feature_normalization = False

    # Compute features
    stream = nm.Stream(
        settings=settings,
        channels=channels,
        verbose=False,
        sfreq=2000,
        line_noise=50
    )
    features = stream.run(data=data_flat, out_dir="features", experiment_name=f"subject_{subject}_{normalize}_{referencing}_split_{n_split}")

    # Reshape features if one epoch is split into more feature samples in the time domain
    if n_split > 1:

        # Reshape the dataframe such that features from one epoch are in one row
        time = features.iloc[::n_split, -1].values
        columns_old = features.columns[:-1]
        features = pd.DataFrame(features.values[:, :-1].reshape(-1, int((features.shape[-1]-1) * n_split)))

        # Add the feature names
        new_columns = []
        for i in range(n_split):
            new_columns.extend([f"{col[1]}_{i}" for col in enumerate(columns_old)])
        features.columns = new_columns

        # Add the time column
        features['time'] = time

        # Save reshaped features as csv
        features.to_csv(f"features\\subject_{subject}_{normalize}_{referencing}_split_{n_split}\\subject_{subject}_{normalize}_{referencing}_split_{n_split}_FEATURES.csv", index=False)

    if normalize:

        # Get epochs data of baseline period (fixation cross)
        data = epochs.get_data(tmin=0, tmax=1)
        n_epochs, n_channels, n_samples = data.shape
        data_flat = data.reshape(n_channels, n_epochs * n_samples)

        # Change segment length
        settings.segment_length_features_ms = 1000

        # Compute features
        stream = nm.Stream(
            settings=settings,
            channels=channels,
            verbose=False,
            sfreq=2000,
            line_noise=50
        )
        features_baseline = stream.run(data=data_flat, out_dir="features", experiment_name=f"subject_{subject}_{normalize}_{referencing}_baseline")

        # Reshape baseline features if one epoch is split into more feature samples in the time domain
        # Duplicate the features
        if n_split > 1:
            features_baseline = pd.concat([features_baseline.iloc[:, :-1]] * n_split, axis=1)
            features_baseline.columns = features.columns[:-1]

        # Normalize
        if norm_method == "percent":
            features_norm = ((features.iloc[:, :-1] - features_baseline) / features_baseline) * 100
            # Append time (un-normalized)
            features_norm['time'] = features['time']

        # Save csv
        features_norm.to_csv(f"features\\subject_{subject}_{normalize}_{referencing}_split_{n_split}\\subject_{subject}_{normalize}_{referencing}_split_{n_split}_FEATURES.csv", index=False)



