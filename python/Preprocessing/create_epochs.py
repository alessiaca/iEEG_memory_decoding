# Restructure data such that each file is transformed into a epochs .fif file

import pandas as pd
import numpy as np
import mne
import os
from mne_bids import find_matching_paths

subjects = np.arange(1, 16)
root = "C:\\Users\\ICN\\OneDrive - Charité - Universitätsmedizin Berlin\\DATA_USZ_NCH"
files = os.listdir(root)

for i, subject in enumerate(subjects):

    subject_name = f'0{subject}' if subject < 10 else f'{subject}'
    bids_paths = find_matching_paths(root, subjects=subject_name, datatypes='ieeg', extensions='.edf')

    # Loop over session files
    for j, bids_path in enumerate(bids_paths[:4]):

        try:

            # Load the ieeg data
            raw = mne.io.read_raw_edf(bids_path.fpath, preload=True)
            sfreq = raw.info['sfreq']
            ch_names = raw.ch_names

            # Load the events
            events_df = pd.read_csv(str(bids_path.fpath)[:-8]+"events.tsv", sep="\t")

            # Create MNE events array
            events = np.array([[onset, 0, 0] for onset in events_df['begSample']])

            # Create epochs
            tmin = 0
            tmax = 7.999
            epochs = mne.Epochs(raw, events, event_id=None, tmin=tmin, tmax=tmax, baseline=None, preload=True)

            # Get the unique electrode groups
            electrode_groups = set([ch[:-1] for ch in ch_names])

            # Remove if less than 8 channels in the group
            channels_to_remove = []
            for group in electrode_groups:
                group_channels = [ch for ch in ch_names if ch.startswith(group)]
                if len(group_channels) < 7:
                    channels_to_remove.extend(group_channels)
                    print(f"Removing group {group} for Subject {subject} due to insufficient channels.")
            # Remove electrodes of different shape
            if subject == 10:
                channels_to_remove.extend(ch_names[:-16])
            if subject == 12:
                channels_to_remove.extend([ch for ch in ch_names if ch.startswith("mLL")])
            if subject == 13:
                channels_to_remove.extend([ch for ch in ch_names if ch.startswith("HIR")])
            if len(channels_to_remove) > 0:
                epochs.drop_channels(channels_to_remove)

                        
            # Save epochs
            epochs.save(f"..\..\..\Data_processed\Original\Data_Subject_{subject}_Session_0{j+1}.fif", overwrite=True)
            events_df.to_csv(f"..\..\..\Data_processed\Original\Data_Subject_{subject}_Session_0{j+1}.csv", index=False)

            # Save the updated electrode locations (after removing channels)
            electrodes_df = pd.read_csv(str(bids_path.fpath)[:-8]+"electrodes.tsv", sep="\t")
            electrodes_df = electrodes_df[~electrodes_df['name'].isin(channels_to_remove)]
            electrodes_df.to_csv(f"..\..\..\Data_processed\Metadata\Subject_{subject}_electrode_locations.csv", index=False)

        except Exception as e:
            print(f"Error processing Subject 0{subject} Session 0{j+1}: {e}")
            pass