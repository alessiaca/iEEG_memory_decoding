# Test which regions of the TFR are significantly different across conditions

import py_neuromodulation as nm
import numpy as np
import os
import mne
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from utils import compute_p_tfr, window_average
from statsmodels.stats.multitest import fdrcorrection
matplotlib.use("Qt5Agg")

reference_scheme = "car"
subjects = np.arange(1, 10)
freqs = np.arange(4, 110, 0.5)

for subject in subjects:

    # Load the epochs
    epochs = mne.read_epochs(f"..\data_epochs\\re_referenced\Data_Subject_0{subject}_{reference_scheme}.fif")
    ch_names = epochs.info["ch_names"]
    epochs.resample(400)

    # Load the set size info
    df = pd.read_csv(f'..\data_epochs\merged\Data_Subject_0{subject}.csv')
    set_size = df["Set size"].to_numpy()
    sizes = np.unique(set_size)

    # Get the index of each set size
    idx_4 = np.where(set_size == 4)[0]
    idx_6 = np.where(set_size == 6)[0]
    idx_8 = np.where(set_size == 8)[0]

    # Select channels of interest (
    if reference_scheme == "bip":
        # every 7th for bip
        chs = [ch_names[i] for i in np.arange(0, len(ch_names), 7)]
    else:
        # every 8th for car
        chs = [ch_names[i] for i in np.arange(0, len(ch_names), 8)]

    for ch in chs:

        # Compute the TFR
        tfr = epochs.compute_tfr(picks=ch, average=False, freqs=freqs, method="morlet", n_cycles=7)
        tfr_norm = tfr.copy().apply_baseline(mode="percent", baseline=(3, 6))

        # Plot the difference
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        tfr_diff = tfr_norm[idx_4].average() - tfr_norm[idx_8].average()
        tfr_diff.plot(tmin=1, tmax=6, axes=axes[0], show=False)
        axes[0].set_title("Difference_4-8")

        # Compute the p_value for each entry in the TFR plot (normalized)
        tfr_array = tfr.get_data(tmin=1, tmax=6)

        # Average over time to reduce dimensionality
        window_size = int(tfr.info["sfreq"] * 0.1)
        tfr_array_small = np.apply_along_axis(lambda x: window_average(x, window_size), axis=-1, arr=tfr_array)

        p_values = compute_p_tfr(tfr_array_small, idx_4, idx_8)

        # FDR correct the p_values
        #rejected, p_values_corrected = fdrcorrection(p_values.flatten(), alpha=0.05, method='indep', is_sorted=False)
        #p_values = p_values_corrected.reshape(p_values.shape)

        # Plot the p_values
        cax = axes[1].imshow(p_values, aspect='auto', origin='lower', cmap='RdBu_r', vmin=0, vmax=0.01)
        cbar = fig.colorbar(cax, ax=axes[1])
        cbar.set_label("P-value")

        # Set axis tick labels
        yticks = axes[1].get_yticks()
        axes[1].set_yticklabels([int(np.min(freqs) + y/2) for y in yticks])
        axes[1].set_xticks([])

        # Add line seperating encoding and maintanance
        axes[1].axvline(x=p_values.shape[-1]*(2/5), color='black', linestyle='--')
        axes[0].axvline(x=3, color='black', linestyle='--')

        # Adjust figure
        plt.subplots_adjust(wspace=0.3)
        plt.suptitle(f"Subject: 0{subject}, Channel: {ch}, difference: 4-8",)

        # Save the figure
        fig.savefig(f"..\\figures\\tfr_subject_0{subject}_{ch}_stats_4_8_{reference_scheme}.png")

        plt.close()

        #plt.show()