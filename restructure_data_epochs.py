# Restructure data such that each file is transformed into a epochs .fif file

import pandas as pd
import h5py
import numpy as np
import mne

subjects = np.arange(1, 10)
n_trials = 50

for i, subject in enumerate(subjects):

    for session in range(1, 8):

        try:
            filename = f"..\data_nix\Data_Subject_0{subject}_Session_0{session}.h5"
            f = h5py.File(filename,'r')

            # Get the iEEG data
            data = f['data'][f'Data_Subject_0{subject}_Session_0{session}']['data_arrays']
            iEEG_data = []

            # Get the task properties
            info = f['metadata']['Session']['sections']['Trial properties']['sections']
            trial_properties = np.zeros((n_trials, 5))

            # Loop over trials
            for j in range(n_trials):

                trial_number = f'0{j+1}' if j < 9 else f'{j+1}'

                try:
                    # Append the iEEG data
                    iEEG_data.append(data[f'iEEG_Data_Trial_{trial_number}']['data'])

                    # Save the trial properties
                    trial_properties[j, 0] = info[f'Trial_{trial_number}']['properties']['Set size'][:][0][0]
                    trial_properties[j, 1] = info[f'Trial_{trial_number}']['properties']['Match'][:][0][0]
                    trial_properties[j, 2] = info[f'Trial_{trial_number}']['properties']['Correct'][:][0][0]
                    trial_properties[j, 3] = info[f'Trial_{trial_number}']['properties']['Response'][:][0][0]
                    if trial_properties[j, 3] > 50:
                        trial_properties[j, 3] -= 50
                    trial_properties[j, 4] = info[f'Trial_{trial_number}']['properties']['Response time'][:][0][0]

                except Exception as e:
                    print(e)

            # Convert to array
            iEEG_data = np.array(iEEG_data)[:, :, 4000:10000]

            # Generate the info
            n_channels = iEEG_data.shape[1]
            ch_names = [f"CH{i}" for i in range(n_channels)]  # Channel names
            ch_types = ["dbs"] * n_channels
            info = mne.create_info(ch_names=ch_names, sfreq=2000, ch_types=ch_types)

            # Create epochs
            epochs = mne.EpochsArray(iEEG_data, info)

            # Save epochs file
            epochs.save(f"..\data_epochs\Data_Subject_0{subject}_Session_0{session}.fif", overwrite=True)

            # Save trial properties as dataframe
            columns = ['Set size', 'Match', 'Correct', 'Response', 'Response time']
            df = pd.DataFrame(trial_properties, columns=columns)
            df.to_csv(f"..\data_epochs\Data_Subject_0{subject}_Session_0{session}.csv", index=False)

        except Exception as e:
            print(e)
            pass

    # Save the electrode locations in MNI space for each patient
    MNI_coordinates = data['iEEG_Electrode_MNI_Coordinates']['data'][:]
    df = pd.DataFrame(MNI_coordinates, columns=['X', 'Y', 'Z'])
    df.to_csv(f"..\data_epochs\Subject_0{subject}_electrode_locations.csv", index=False)

    print("h")
