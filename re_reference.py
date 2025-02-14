# Re-reference the data
# Bipolarly in ascending order
# CAR within macroelectrode leads (8 channels per electrode)

import py_neuromodulation as nm
import numpy as np
import os
import mne
from utils import compute_bipolar_epochs, compute_car_epochs

subjects = np.arange(1, 10)

for subject in subjects:

    # Load the epochs
    epochs = mne.read_epochs(f"..\data_epochs\original\Data_Subject_0{subject}_Session_01.fif")
    ch_names = epochs.info["ch_names"]

    # Define bipolar reference scheme (list of electrode pairs to be subtracted)
    reference = []
    for x in range(len(ch_names) - 1):
        if x not in np.arange(7, len(ch_names), 8):
            reference.append([ch_names[x], ch_names[x+1]])
    # Compute bipolar epochs
    bipolar_epochs = compute_bipolar_epochs(epochs, reference)

    # Save
    bipolar_epochs.save(f'..\data_epochs\\re_referenced\Data_Subject_0{subject}_Session_01_bip.fif', overwrite=True)

    # Define CAR scheme
    groups = [ch_names[i:i + 8] for i in range(0, len(ch_names), 8)]
    car_epochs = compute_car_epochs(epochs, groups)

    # Save
    car_epochs.save(f'..\data_epochs\\re_referenced\Data_Subject_0{subject}_Session_01_car.fif', overwrite=True)