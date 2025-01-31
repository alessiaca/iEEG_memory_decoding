# Merge data from sessions for each subject

import pandas as pd
import numpy as np
import mne
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

subjects = np.arange(1, 10)

# Get a list of fif datasets in "..\data_epochs\
files = os.listdir("..\data_epochs")

for subject in subjects:

    fif_files = [file for file in files if file.endswith(".fif") and file.startswith(f"Data_Subject_0{subject}")]
    csv_files = [file for file in files if file.endswith(".csv") and file.startswith(f"Data_Subject_0{subject}")]

    epochs_all = []
    df_all = []

    # Loop over all files
    for j, (fif_file, csv_file) in enumerate(zip(fif_files, csv_files)):

        # Load the epochs
        epochs = mne.read_epochs(f"..\data_epochs\{fif_file}")
        epochs_all.append(epochs)

        # Load csv
        df = pd.read_csv(f"..\data_epochs\{csv_file}")
        df_all.append(df)

    # Merge epoch objects and save
    epochs_combined = mne.concatenate_epochs(epochs_all)
    epochs_combined.save(f'..\data_epochs\Data_Subject_0{subject}.fif', overwrite=True)

    # Merge df and save
    df_combined = pd.concat(df_all, ignore_index=True)
    df_combined.to_csv(f'..\data_epochs\Data_Subject_0{subject}.csv', index=False)






