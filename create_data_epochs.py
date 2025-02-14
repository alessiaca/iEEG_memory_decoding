# Restructure data such that each file is transformed into a epochs .fif file

import pandas as pd
import h5py
import numpy as np
import mne
import os

subjects = np.arange(1, 10)
n_trials = 50
root = "..\data_nix"
files = os.listdir(root)

for i, subject in enumerate(subjects):


    # Get all session files
    session_files = [file for file in files if file.endswith(".h5") and file.startswith(f"Data_Subject_0{subject}")]

    for j, file in enumerate(session_files):

        f = h5py.File(root + '\\' + file,'r')

        # Get the iEEG data
        data = f['data'][f'Data_Subject_0{subject}_Session_0{j+1}']['data_arrays']
        iEEG_data = []

        # Get the task properties
        info = f['metadata']['Session']['sections']['Trial properties']['sections']
        trial_properties = []

        # Loop over trials
        for k in range(n_trials):

            trial_number = f'0{k+1}' if k < 9 else f'{k+1}'

            if f'Trial_{trial_number}' in info:

                # Get the data for a trial
                set_size = info[f'Trial_{trial_number}']['properties']['Set size'][:][0][0]
                match = info[f'Trial_{trial_number}']['properties']['Match'][:][0][0]
                correct = info[f'Trial_{trial_number}']['properties']['Correct'][:][0][0]
                response = info[f'Trial_{trial_number}']['properties']['Response'][:][0][0]
                response_time = info[f'Trial_{trial_number}']['properties']['Response time'][:][0][0]
                iEEG_trial_data = data[f'iEEG_Data_Trial_{trial_number}']['data']

                # Append the data
                iEEG_data.append(iEEG_trial_data)
                trial_properties.append([set_size, match, correct, response, response_time])

        # Convert to array
        iEEG_data = np.array(iEEG_data)

        # Generate the info
        n_channels = iEEG_data.shape[1]
        ch_names = [f"CH{i}" for i in range(n_channels)]  # Channel names
        ch_types = ["dbs"] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=2000, ch_types=ch_types)

        # Create epochs
        epochs = mne.EpochsArray(iEEG_data, info)

        # Save epochs file
        epochs.save(f"..\data_epochs\original\Data_Subject_0{subject}_Session_0{j+1}.fif", overwrite=True)

        # Save trial properties as dataframe
        columns = ['Set size', 'Match', 'Correct', 'Response', 'Response time']
        df = pd.DataFrame(trial_properties, columns=columns)
        df.to_csv(f"..\data_epochs\original\Data_Subject_0{subject}_Session_0{j+1}.csv", index=False)

    # Save the electrode locations in MNI space for each patient
    MNI_coordinates = data['iEEG_Electrode_MNI_Coordinates']['data'][:]
    df = pd.DataFrame(MNI_coordinates, columns=['X', 'Y', 'Z'])
    df.to_csv(f"..\data_epochs\metadata\Subject_0{subject}_electrode_locations.csv", index=False)

