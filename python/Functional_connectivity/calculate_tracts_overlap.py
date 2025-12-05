import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr
import sqlite3
from scipy.io import loadmat
from itertools import product
from joblib import Parallel, delayed
from dipy.io.stateful_tractogram import StatefulTractogram, Space
import os
from dipy.tracking.utils import target

# Load decoding accuracies
df_decoding = pd.read_csv(f"../Decoding/decoding_accuracies.csv")
flipped = "false"

# Load the grey matter mask
gm_path = "../../../Atlas/rc1mask.nii"
image_c1 = nib.load(gm_path).get_fdata()

overlap = []
# Loop over each subject and channel
for _, row in df_decoding.iterrows():
    subject = int(row["subject"])
    channel = row["channel"]

    # Load the electrode ROI nii files
    path = f"../../../Electrode_nifti/Subject_{subject}_bipolar_{channel}_flipped_{flipped}.nii"
    image = nib.load(path).get_fdata()

    # Overlap with grey matter
    n_overlap = ((image_c1 > 0.5) & (image > 0.5)).sum()
    n_roi = (image > 0.5).sum()
    percent_overlap = n_overlap / n_roi if n_roi > 0 else 0
    overlap.append([subject, channel, percent_overlap])

# Save csv of overlap
overlap_df = pd.DataFrame(overlap, columns=["subject", "channel", "percent_overlap"])
overlap_df.to_csv(f"grey_matter_overlap.csv", index=False)

# Plot as violin plot per patient
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.violinplot(x="subject", y="percent_overlap", data=overlap_df, inner=None)
sns.scatterplot(x="subject", y="percent_overlap", data=overlap_df, color='k', alpha=0.6)
plt.title("Distribution of Grey Matter Overlap per Subject")
plt.xlabel("Subject")
plt.ylabel("Percent Overlap")
plt.show()