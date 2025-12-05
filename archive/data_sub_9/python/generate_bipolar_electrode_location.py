import mne
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import savemat
import warnings
warnings.filterwarnings("ignore")


# Subjects
subjects = np.arange(1, 10)

for subject in subjects:

    mni_locations = pd.read_csv(f"../../Processed_data/metadata/Subject_0{subject}_electrode_locations.csv")
    n_channels = len(mni_locations)
    bipolar_locations = []

    overlap = pd.read_csv(f"../../Processed_data/metadata/Subject_0{subject}_overlap.csv")
    bipolar_overlap = []

    for i in range(n_channels - 1):
        if i not in np.arange(7, n_channels, 8):
            bipolar_locations.append(mni_locations.to_numpy()[i:i+2].mean(axis=0))
            bipolar_overlap.append(overlap.to_numpy()[i:i+2].mean(axis=0))

    # Save csv of overlap
    bipolar_locations_df = pd.DataFrame(bipolar_locations, columns=["X", "Y", "Z"])
    bipolar_locations_df.to_csv(f"../../Processed_data/metadata/Subject_0{subject}_electrode_locations_bip.csv", ignore_index=True)
