# Re-reference the data bipolarly
import numpy as np
import mne
import pandas as pd

def compute_bipolar_epochs(epochs, channel_pairs):

    # Create a dictionary to store new bipolar signals
    bipolar_data = []
    bipolar_ch_names = []
    for ch1, ch2 in channel_pairs:
        # Compute bipolar signal
        bipolar_signal = epochs.get_data(picks=ch1) - epochs.get_data(picks=ch2)
        # Define new channel name
        new_channel = ch1
        # Append new data
        bipolar_data.append(bipolar_signal)
        bipolar_ch_names.append(new_channel)

    # Create a new info object for bipolar channels
    new_info = mne.create_info(
        ch_names=bipolar_ch_names,
        sfreq=epochs.info['sfreq'],
        ch_types='dbs'
    )

    # Convert bipolar data to MNE array
    bipolar_data = np.array(bipolar_data).squeeze().transpose(1, 0, 2)
    bipolar_epochs = mne.EpochsArray(
        data=bipolar_data,
        info=new_info,
        events=epochs.events,
        event_id=epochs.event_id,
        tmin=epochs.tmin
    )

    return bipolar_epochs


subjects = np.arange(1, 16)

for subject in subjects:

    # Load the epochs
    epochs = mne.read_epochs(f"..\..\..\Data_processed\Merged\Data_Subject_{subject}.fif")
    ch_names = epochs.info["ch_names"]

    # Define bipolar reference scheme (list of electrode pairs to be subtracted)
    pairs = []
    for i_chan, chan in enumerate(ch_names[:-1]):
        group_chan = chan[:-1]
        next_chan = ch_names[i_chan+1]
        if next_chan[:-1] == group_chan:
            pairs.append([ch_names[i_chan], ch_names[i_chan+1]])
    # Compute bipolar epochs
    bipolar_epochs = compute_bipolar_epochs(epochs, pairs)

    # Save
    bipolar_epochs.save(f"..\..\..\Data_processed\\Re_referenced\Data_Subject_{subject}.fif", overwrite=True)

    # Compute the average bipolar locations
    electrodes_df = pd.read_csv(f"..\..\..\Data_processed\Metadata\Subject_{subject}_electrode_locations.csv")

    # Create new dataframe for bipolar electrode locations
    electrodes_bipolar = []
    for pair in pairs:
        elec1 = electrodes_df[electrodes_df['name'] == pair[0]].iloc[0]
        elec2 = electrodes_df[electrodes_df['name'] == pair[1]].iloc[0]
        avg_loc = (elec1[['x','y','z']].values + elec2[['x','y','z']].values) / 2
        electrodes_bipolar.append({
            'bipolar_name': f"{pair[0]}",
            'X': avg_loc[0],
            'Y': avg_loc[1],
            'Z': avg_loc[2]
        })

    electrodes_bipolar_df = pd.DataFrame(electrodes_bipolar)

    # --- Save bipolar electrode locations ---
    electrodes_bipolar_df.to_csv(f"..\\..\\..\\Data_processed\\Metadata\\Subject_{subject}_bipolar_electrode_locations.csv", index=False)

    # Save behavioral data unchanged
    df = pd.read_csv(f"..\..\..\Data_processed\Merged\Data_Subject_{subject}.csv")
    df.to_csv(f"..\\..\\..\\Data_processed\\Re_referenced\\Data_Subject_{subject}.csv", index=False)
