# Inspect the data:

import pandas as pd
import numpy as np
import mne
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

n_trials = 50
subjects = np.arange(1, 10)

# Get a list of fif datasets in "..\data_epochs\
files = os.listdir("..\data_epochs")

for subject in subjects:

    fif_files = [file for file in files if file.endswith(".fif") and file.startswith(f"Data_Subject_0{subject}")]

    fig, axes = plt.subplots(len(fif_files), 1)

    # Loop over all files
    for j, file in enumerate(fif_files):

        # Load the epochs
        epochs = mne.read_epochs(f"..\data_epochs\{file}")

        psd = epochs.compute_psd(fmin=0, fmax=50)
        psd.plot(average=False, axes=axes[j])

    # Save
    fig.savefig(f"..\\figures\\{subject}.png")


#plt.show()




