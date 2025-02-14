# Inspect the TFR

import py_neuromodulation as nm
import numpy as np
import os
import mne
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")

reference_scheme = "car"
subjects = np.arange(1, 10)
freqs = np.arange(4, 150, 2)

for subject in subjects:

    # Load the epochs
    epochs = mne.read_epochs(f"..\data_epochs\\re_referenced\Data_Subject_0{subject}_{reference_scheme}.fif")
    ch_names = epochs.info["ch_names"]
    #epochs.resample(500)

    # Load the set size info
    df = pd.read_csv(f'..\data_epochs\merged\Data_Subject_0{subject}.csv')
    set_size = df["Set size"].to_numpy()
    sizes = np.unique(set_size)

    # Select channels of interest (
    if reference_scheme == "bip":
        # every 7th for bip
        chs = [ch_names[i] for i in np.arange(0, len(ch_names), 7)]
    else:
        # every 8th for car
        chs = [ch_names[i] for i in np.arange(0, len(ch_names), 8)]

    for ch in chs:

        # Compute the average TFR
        tfr = epochs.compute_tfr(picks=ch, average=False, freqs=freqs, method="morlet", n_cycles=freqs/2)

        # Apply baseline
        #tfr.apply_baseline(mode="zscore", baseline=(None, None))

        fig, axes = plt.subplots(1, len(sizes)+1, figsize=(15, 5))
        tfr_sizes = []
        for i, size in enumerate(sizes):

            idx = np.where(set_size == size)[0]
            tfr_size = tfr[idx].average().apply_baseline(mode="zscore", baseline=(None, None))
            tfr_size.plot(vmin=-4, vmax=4, tmin=0.5, tmax=7.5, axes=axes[i], show=False)

            axes[i].set_title(f"Set size: {size}")
            tfr_sizes.append(tfr_size)

        # Plot the difference
        tfr_diff = tfr_sizes[0] - tfr_sizes[2]
        tfr_diff.plot(vmin=-4, vmax=4, tmin=0.5, tmax=7.5, axes=axes[-1], show=False)
        axes[-1].set_title("Difference_4-8")

        # Save the figure
        fig.savefig(f"..\\figures\\tfr_subject_0{subject}_{ch}.png")

        plt.close()