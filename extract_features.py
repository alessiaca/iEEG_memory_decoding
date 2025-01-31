# Extract features for each subject

import py_neuromodulation as nm
import numpy as np
import os
import mne

subjects = np.arange(1, 10)
files = os.listdir("..\data_epochs")

# PN settings
settings = nm.NMSettings.get_fast_compute()
settings.features.fft = True
settings.features.return_raw = True
settings.features.raw_hjorth = True
settings.sampling_rate_features_hz = 0.3
settings.segment_length_features_ms = 3000
settings.postprocessing.feature_normalization = False


for subject in subjects:

    # Load the epochs
    epochs = mne.read_epochs(f"..\data_epochs\Data_Subject_0{subject}.fif")

    ch_names = epochs.info["ch_names"]
    ch_types = epochs.get_channel_types()

    channels = nm.utils.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference=None,
        used_types=ch_types,
        target_keywords=None,
    )

    # Compute features
    stream = nm.Stream(
        settings=settings,
        channels=channels,
        verbose=False,
        sfreq=2000,
        line_noise=50
    )

    # Get epochs data
    data = epochs.get_data()
    # Reshape
    data_flat = data.reshape(data.shape[1], data.shape[0] * data.shape[2])
    stream.run(data=data_flat, out_dir="features", experiment_name=f"subject_{subject}")
