# Extract features for each subject

import py_neuromodulation as nm
import numpy as np
import os
import mne

subjects = np.arange(1, 10)

# PN settings
settings = nm.NMSettings.get_fast_compute()
settings.features.fft = True
settings.features.welch = False
settings.features.return_raw = False
settings.features.raw_hjorth = False
settings.sampling_rate_features_hz = 0.3
settings.segment_length_features_ms = 1000
settings.postprocessing.feature_normalization = False
# Delete frequency ranges
del settings.frequency_ranges_hz["HFA"]
del settings.frequency_ranges_hz["high_gamma"]
del settings.frequency_ranges_hz["low_gamma"]
del settings.frequency_ranges_hz["high_beta"]


for subject in subjects:

    # Load the epochs
    epochs = mne.read_epochs(f"..\data_epochs\\re_referenced\Data_Subject_0{subject}_car.fif")

    ch_names = epochs.info["ch_names"]
    ch_types = epochs.get_channel_types()

    channels = nm.utils.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        used_types=ch_types,
        reference=None,
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
    data = epochs.get_data(tmin=3, tmax=6)
    # Reshape
    data_flat = data.reshape(data.shape[1], data.shape[0] * data.shape[2])
    stream.run(data=data_flat, out_dir="features", experiment_name=f"subject_{subject}")
