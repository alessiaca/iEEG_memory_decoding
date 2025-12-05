import mne
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import savemat
import warnings
warnings.filterwarnings("ignore")

# Subjects
subjects = np.arange(1, 10)

# Load grey matter map
gm_path = "../../Atlas/rc1mask.nii"
image_c1 = nib.load(gm_path).get_fdata()

# Dictionary store the % overlap of each channel with grey matter
overlap = []

for subject in subjects:
    epochs = mne.read_epochs(f"../../Processed_data/merged/Data_Subject_0{subject}.fif", preload=False, verbose=False)
    channels = epochs.info["ch_names"]
    
    bad_ch_subject = []
    for channel in channels:
        # Path to the electrode nifti
        path = f"../../Electrode_nifti/Subject_{subject}_none_{int(channel[2:])+1}_flipped_false.nii"
        image = nib.load(path).get_fdata()

        # Overlap with grey matter
        n_overlap = ((image_c1 > 0.5) & (image > 0.5)).sum()
        n_roi = (image > 0.5).sum()
        percent_overlap = n_overlap / n_roi if n_roi > 0 else 0
        overlap.append([subject, int(channel[2:])+1, percent_overlap])

# Save csv of overlap
overlap_df = pd.DataFrame(overlap, columns=["subject", "channel", "percent_overlap"])
overlap_df.to_csv(f"../../Processed_data/metadata/grey_matter_overlap.csv", index=False)
