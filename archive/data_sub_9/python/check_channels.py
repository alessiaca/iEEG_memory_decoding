import mne
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_welch
import numpy as np
from scipy.io import savemat
import pandas as pd
import nibabel as nib


# Exclude electrodes whoch are outside of the grey matter (use map?)
# Check exemplar data and psd for each channel to make sure the data is valid
subjects = np.arange(1, 10)
bad_channels = {
    "1":[7, 15, 40, 41, 42, 43, 44, 45, 46, 47],
    "2":[7, 23, 39, 55],
    "3":[31],
    "4":[24, 25,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    "5":[16,17,18,8,9,25,26,27,28,29,30,31],
    "6":[0, 55],
    "7":[8, 56],
    "8":[1,2,3,4,5,6,7,48,49,50,51,52,53,54, 39], 
    "9":[30, 31],
}
bc_dict = {f'subject_{k}': v for k, v in bad_channels.items()}
savemat("../../Processed_data/metadata/bad_channels.mat", {"bad_channels": bc_dict})

def on_pick(event):
    line = event.artist
    label = line.get_label()
    print(f'You clicked on: {label}')

for subject in subjects:
    epochs = mne.read_epochs(f"../../Processed_data/merged/Data_Subject_0{subject}.fif")
    channels = epochs.info["ch_names"]

    file_path = f"../../Processed_data/metadata/subject_{subject}_overlap.csv"
    overlap_df = pd.read_csv(file_path)

    fig, axs = plt.subplots(1, 3, figsize=(19, 3))
    for channel in channels:
        fig.suptitle(f"Subject {subject}")

        if overlap_df[overlap_df["channel"] == channel]["percent_overlap"].values[0] < 0.25:
            color = "red"
        else:
            color = "blue"

        # Plot raw data of last epoch
        data = epochs[-1].get_data(picks=channel).squeeze() # First epoch, first 7 channels
        axs[0].plot(data[:2000], color= color)
        axs[0].set_title(f"Raw data last epoch")

        # Compute PSD for the last epoch
        sfreq = epochs.info['sfreq']
        psd, freqs = psd_array_welch(data, sfreq=sfreq, fmin=0.5, fmax=40.0, n_fft=2000)
        axs[1].plot(freqs, np.log(psd), picker=True, label=channel, color=color)
        axs[1].set_title(f"PSD last epoch")

        # Compute PSD for the all epochs
        data = epochs[:100].get_data(picks=channel)
        sfreq = epochs.info['sfreq']
        psd, freqs = psd_array_welch(data, sfreq=sfreq, fmin=0.5, fmax=40.0, n_fft=2000)
        psd_mean = psd.mean(axis=0).squeeze()
        axs[2].plot(freqs, np.log(psd_mean), picker=True, label=channel, color=color)
        axs[2].set_title(f"PSD last epoch")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()
