# Merge data from sessions for each subject

import pandas as pd
import numpy as np
import mne
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

subjects = np.arange(1, 16)

# Get a list of fif datasets in "..\data_epochs\
root_original = "..\..\..\Data_processed\Original"
root_merged = "..\..\..\Data_processed\Merged"
files = os.listdir(root_original)

for subject in subjects:

    fif_files = [file for file in files if file.endswith(".fif") and file.startswith(f"Data_Subject_{subject}_")]
    csv_files = [file for file in files if file.endswith(".csv") and file.startswith(f"Data_Subject_{subject}_")]

    epochs_all = []
    df_all = []

    # Loop over all files
    for j, (fif_file, csv_file) in enumerate(zip(fif_files, csv_files)):

        # Load the epochs
        epochs = mne.read_epochs(f"{root_original}\\{fif_file}")
        epochs_all.append(epochs)

        # Load csv
        df = pd.read_csv(f"{root_original}\\{csv_file}")
        df_all.append(df)

    # Merge epoch objects and save
    print(subject)
    epochs_combined = mne.concatenate_epochs(epochs_all)
    epochs_combined.save(f'{root_merged}\\Data_Subject_{subject}.fif', overwrite=True)

    # Merge df and save
    df_combined = pd.concat(df_all, ignore_index=True)
    df_combined.to_csv(f'{root_merged}\\Data_Subject_{subject}.csv', index=False)





