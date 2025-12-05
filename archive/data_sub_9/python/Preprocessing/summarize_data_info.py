import pandas as pd
import h5py
import numpy as np


subjects = np.arange(1, 10)
cohort_info = np.zeros((len(subjects), 4))

for i, subject in enumerate(subjects):

    trial_number = []
    for session in range(1, 10):

        try:
            filename = f"..\data_nix\Data_Subject_0{subject}_Session_0{session}.h5"
            f = h5py.File(filename,'r')

            # Count number of sessions
            cohort_info[i, 0] += 1

            # Get the maximum and minimum trial number across session
            data = f['data'][f'Data_Subject_0{subject}_Session_0{session}']
            data_array = data['data_arrays']
            # Count keys with iEEG in the name
            iEEG_trials = [key for key in data_array.keys() if 'iEEG' in key]
            trial_number.append(len(iEEG_trials))
        except:
            pass

        cohort_info[i, 1] = np.min(trial_number)
        cohort_info[i, 2] = np.max(trial_number)

        # Get the number of iEEG channels
        cohort_info[i, 3]= data_array[iEEG_trials[0]]['data'].shape[0]

# Transform to dataframe with patient id
cohort_info = np.insert(cohort_info, 0, subjects, axis=1)
cohort_info = pd.DataFrame(cohort_info, columns=['Subject', 'Sessions', 'Min_Trial', 'Max_Trial', 'iEEG_Channels'])

print(cohort_info)
