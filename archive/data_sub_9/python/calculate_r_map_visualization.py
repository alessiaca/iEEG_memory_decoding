import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr
import sqlite3
from itertools import product
from joblib import Parallel, delayed
import os


# Set parameters
referencing = "none"
Hz = 5
length = 1000
n_runs = 200
model_name = f"{referencing}_Hz_{Hz}_length_{length}_runs_{n_runs}"
mode = "Functional"

# Options to loop over
subject_mode = np.array([2, 3, 5, 6, 7, 8, 9])#  [[3, 6, 7, 8, 9], np.arange(1, 10), np.arange(2, 10)]
flip_mode = "true"
overlap_thresh = None
mask = None # ["memory_association-test_z_FDR_0.01", None, "rFornix", "rnucleus_accumbens", "rc1mask"]

# Load input data
decoding_df = pd.read_csv(f"../../Decoding_results/{model_name}.csv")
overlap_df = pd.read_csv("../../Processed_data/metadata/grey_matter_overlap.csv")


# Get the mask image
if mask:
    mask_image = nib.load(f"{mask}.nii").get_fdata()

# Filter > 50 % decoding performance 
decoding_df = decoding_df[decoding_df["accuracy"] > 0.5]
# Filter subjects of interest
decoding_df = decoding_df[decoding_df["subject"].isin(subject_mode)]
# Filter channels based on grey matter overlap
if overlap_thresh:
    decoding_df= decoding_df[overlap_df["percent_overlap"] > overlap_thresh]

# Load functional connectivity maps
image_all = []
decoding_df_iter = decoding_df.copy()
for i, row in decoding_df_iter.iterrows():
    subject = int(row["subject"])
    channel = int(row["channel"])

    try:
        flip_options = ["true", "false"] if flip_mode == "true" else ["false"]
        for flipped in flip_options:
            #path = f"../../Maps/Functional_connectivity/Subject_{subject}_{referencing}_{channel}_flipped_{flipped}_func_seed_AvgR_Fz.nii"
            if mode == "Functional":
                path = f"../../Maps/Functional_connectivity/Subject_{subject}_{referencing}_{channel}_flipped_{flipped}_func_seed_AvgR_Fz.nii"
            if mode == "Structural":
                path = f"../../Maps/Structural_connectivity/sSubject_{subject}_{referencing}_{channel}_flipped_{flipped}_struc_seed.nii"
            if os.path.exists(path):
                break
        image = nib.load(path).get_fdata()
        image_all.append(image)
    except Exception as e:
        decoding_df = decoding_df.drop(i, axis=0)

image_all = np.array(image_all)
decoding_array = decoding_df["accuracy"].to_numpy()

# Calculate the optimal r-map on the remaining patients
r_map = np.zeros((image_all[0].shape[0], image_all[0].shape[1], image_all[0].shape[2], 2))

for j in range(image_all[0].shape[0]):
    for k in range(image_all[0].shape[1]):
        for l in range(image_all[0].shape[2]):
            x = image_all[:, j, k, l]
            y = decoding_array
            # Mask out NaNs
            valid = ~np.isnan(x) & ~np.isnan(y)
            r_map[j, k, l, 0], r_map[j, k, l, 1] = spearmanr(x[valid], y[valid])

# Save r and p map in nii format
template_path = f"../../Maps/Functional_connectivity/Subject_1_bipolar_1_func_seed_AvgR_Fz.nii"  # Assuming a path for NIfTI
nii = nib.load(template_path)
nii_new = nib.Nifti1Image(r_map[..., 0], nii.affine, nii.header)
new_path = f"../../Maps/Correlation/{model_name}/r_map_{mode}_{flip_mode}_{mask}_{overlap_thresh}_{np.mean(subject_mode)}.nii"
nib.save(nii_new, new_path)
nii_new = nib.Nifti1Image(r_map[..., 1], nii.affine, nii.header)
new_path = f"../../Maps/Correlation/{model_name}/r_map_{mode}_{flip_mode}_{mask}_{overlap_thresh}_{np.mean(subject_mode)}_p.nii"
nib.save(nii_new, new_path)
tmp = r_map[..., 1].copy()
tmp[r_map[..., 1] >= 0.05] = 0
tmp[r_map[..., 1] < 0.05] = 1
nii_new = nib.Nifti1Image(tmp, nii.affine, nii.header)
new_path = f"../../Maps/Correlation/{model_name}/r_map_{mode}_{flip_mode}_{mask}_{overlap_thresh}_{np.mean(subject_mode)}_p_thres.nii"
nib.save(nii_new, new_path)
