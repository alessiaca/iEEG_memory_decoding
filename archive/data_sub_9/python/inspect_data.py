import mne
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_welch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

subjects = np.arange(3, 10)
modes = ["None", "car", "bip", "car_all"]

fig, axs = plt.subplots(4, 2, figsize=(12, 6))

for subject in subjects:
    fig.suptitle(f"Subject {subject}")
    for i, mode in enumerate(modes):
        # Load CAR-referenced data
        if mode == "None":
            epochs = mne.read_epochs(f"../../Processed_data/merged/Data_Subject_0{subject}.fif")
        else:
            epochs = mne.read_epochs(f"../../Processed_data/re_referenced/Data_Subject_0{subject}_{mode}.fif")

        data_car = epochs[0].get_data()[0, :, :]  # First epoch, first 7 channels
        axs[i, 0].plot(data_car[:, :].T)
        axs[i, 0].set_title("CAR - Raw Epoch (Ch 1–7)")

        # Compute PSD for first epoch (channels 1–7)
        sfreq = epochs.info['sfreq']
        psd_car, freqs_car = psd_array_welch(data_car, sfreq=sfreq, fmin=0.5, fmax=40.0, n_fft=2000)
        for ch_psd in psd_car:
            axs[i, 1].plot(freqs_car, np.log(ch_psd))
        axs[i, 1].set_title("CAR - PSD (1st Epoch, Ch 1–7)")
        axs[i, 1].set_xlabel("Frequency (Hz)")
        axs[i, 1].set_ylabel("PSD (uV^2/Hz)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
