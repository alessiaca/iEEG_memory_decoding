import numpy as np
import mne
from numba import njit
from scipy.stats import ttest_ind, mannwhitneyu

def compute_bipolar_epochs(epochs, channel_pairs):

    # Create a dictionary to store new bipolar signals
    bipolar_data = []
    bipolar_ch_names = []
    for ch1, ch2 in channel_pairs:
        # Compute bipolar signal
        bipolar_signal = epochs.get_data(picks=ch1) - epochs.get_data(picks=ch2)
        # Define new channel name
        #new_channel = f"{ch1}-{ch2}"
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


def compute_car_epochs_all(epochs):

    # Compute common average reference
    car_data = epochs.get_data() - np.mean(epochs.get_data(), axis=1, keepdims=True)

    # Store CAR-referenced data and names
    car_ch_names = [name + '_CAR' for name in epochs.info['ch_names']]

    # Create new info structure for CAR-referenced channels
    new_info = mne.create_info(
        ch_names=car_ch_names,
        sfreq=epochs.info['sfreq'],
        ch_types='eeg'
    )

    # Create new Epochs object with only CAR-referenced data
    car_epochs = mne.EpochsArray(
        data=car_data,
        info=new_info,
        events=epochs.events,
        event_id=epochs.event_id,
        tmin=epochs.tmin
    )

    return car_epochs


def compute_car_epochs(epochs, groups):

    car_data = []
    car_ch_names = []

    for group in groups:

        # Get data for the current group
        group_data = epochs.get_data(picks=group)

        # Compute common average reference for the group
        car_signal = group_data - np.mean(group_data, axis=1, keepdims=True)

        # Store CAR-referenced data and names
        car_data.append(car_signal)
        car_ch_names.extend([name + '_CAR' for name in group])

    # Convert list to 3D NumPy array (n_epochs, n_channels, n_times)
    car_data = np.concatenate(car_data, axis=1)  # Merge all groups along channel axis

    # Create new info structure for CAR-referenced channels
    new_info = mne.create_info(
        ch_names=car_ch_names,
        sfreq=epochs.info['sfreq'],
        ch_types='eeg'
    )

    # Create new Epochs object with only CAR-referenced data
    car_epochs = mne.EpochsArray(
        data=car_data,
        info=new_info,
        events=epochs.events,
        event_id=epochs.event_id,
        tmin=epochs.tmin
    )

    return car_epochs

def compute_p_tfr(tfr, idx_1, idx_2):
    tfr = tfr.squeeze()
    _, n_freqs, n_times = tfr.shape
    p_values = np.zeros((n_freqs, n_times))
    for f in range(n_freqs):
        for t in range(n_times):
            p_values[f, t] = mannwhitneyu(tfr[idx_1, f, t].squeeze(), tfr[idx_2, f, t])[1]

    return p_values


def window_average(data, window_size):

    num_windows = len(data) // window_size
    reshaped_data = data[:num_windows * window_size].reshape(-1, window_size)
    averaged_data = reshaped_data.mean(axis=-1)

    return averaged_data
