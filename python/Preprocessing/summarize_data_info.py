import pandas as pd
import mne
import numpy as np

subjects = np.arange(1, 16)
ch_nums = []

for i, subject in enumerate(subjects):

    # Load the csv file 
    df = pd.read_csv(f"..\..\..\Processed_data\merged\Data_Subject_{subject}.csv")
    epochs = mne.read_epochs(f"..\..\..\Processed_data\merged\Data_Subject_{subject}.fif")

    ch_nums.append(len(epochs.ch_names)/8)
    print(epochs.ch_names)

    # Get the minimal number of trials across sessions
    print(subject)
    print(len(epochs))
    print(len(df))
    print()
    #print(df.groupby("SetSize").count())


print("Channel numbers across subjects:")
print(ch_nums)