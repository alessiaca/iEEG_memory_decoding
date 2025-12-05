# Extract features for each subject

import py_neuromodulation as nm
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
import mne
import sys


def extract_features(subject, referencing, Hz, length):

    # Load the epochs
    if referencing == "none":
            epochs = mne.read_epochs(f"../../Processed_data/merged/Data_Subject_0{subject}.fif")
    else:
        epochs = mne.read_epochs(f"../../Processed_data/re_referenced/Data_Subject_0{subject}_{referencing}.fif")


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
    data = epochs[:].get_data(tmin=0, tmax=6)
    n_epochs, n_channels, n_samples = data.shape
    data_flat = data.reshape(n_channels, n_epochs * n_samples)

    # PN settings
    settings = nm.NMSettings.get_fast_compute()
    settings.features.fft = True
    settings.features.welch = False
    settings.features.return_raw = False
    settings.features.raw_hjorth = True
    settings.features.sharpwave_analysis = False
    settings.segment_length_features_ms = length
    settings.preprocessing = []
    settings.sampling_rate_features_hz = Hz
    settings.postprocessing.feature_normalization = False
    del settings.frequency_ranges_hz["HFA"]
    del settings.frequency_ranges_hz["high_gamma"]

    # Compute features
    stream = nm.Stream(
        settings=settings,
        channels=channels,
        verbose=False,
        sfreq=2000,
        line_noise=50
    )
    df = stream.run(data=data_flat, out_dir="", experiment_name="", save_csv=False)


referencing = "bip"
Hz = 5
length = 1000
subjects = np.arange(1, 10)
Parallel(n_jobs=1)(delayed(extract_features)(i, referencing, Hz, length) for i in subjects)
