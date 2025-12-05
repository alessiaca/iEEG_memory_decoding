# R_map calculation across subjects 

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr
from scipy.io import savemat
import os
from joblib import Parallel, delayed

# Define parameters
referencing = "none"
Hz = 5
length = 1000
n_runs = 200
model_name = f"{referencing}_Hz_{Hz}_length_{length}_runs_{n_runs}"
measure = "accuracy"
subjects = np.arange(1, 10) # Exclude the patients who have a condition which is expect to alter the anatomy (making the normative approaches difficult)
flip_mode = "true"
mode = "Functional"

# Load the decoding results
df_decoding = pd.read_csv(f"../../Decoding_results/{model_name}.csv")

# Load the overlap with the grey matter
df_overlap = pd.read_csv(f"../../Processed_data/metadata/grey_matter_overlap.csv")

# Filter the channels which have above-chance level decoding for all runs (with random seeds)
df_decoding = df_decoding[(df_decoding[measure] > 0.5)]

# Filter out the subjects of interest
df_decoding = df_decoding[df_decoding["subject"].isin(subjects)]

# Filter out the channels that overlap with the grey matter at least 25 %
#df_decoding = df_decoding[(df_overlap["percent_overlap"] > 0.1)]

# Load the functional connectivity maps for each subject/channel combination
image_all = []
df_decoding_new = df_decoding.copy()
for i, row in df_decoding_new.iterrows():
    subject = int(row["subject"])
    channel = int(row["channel"])

    flip_options = ["true", "false"] if flip_mode == "true" else ["false"]
    path = None
    try:
        for flipped in flip_options:
            if mode == "Functional":
                path = f"../../Maps/Functional_connectivity/Subject_{subject}_{referencing}_{channel}_flipped_{flipped}_func_seed_AvgR_Fz.nii"
            else:
                path = f"../../Maps/Structural_connectivity/sSubject_{subject}_{referencing}_{channel}_struc_seed.nii"
            if os.path.exists(path):
                break
        image = nib.load(path).get_fdata()
        image_all.append(image)
    except Exception as e:
        print(e)
        df_decoding = df_decoding.drop(i, axis=0)

image_all = np.array(image_all)

# Extract the measure
decoding_measure = df_decoding[measure].to_numpy()

# Function to compute correlation maps for a given subject
def compute_corr_maps(i, image_all, decoding_measure, df_decoding, model_name):
    if i == 0:
        idx_train = np.arange(len(decoding_measure))
    else:
        idx_train = np.where(df_decoding["subject"] != i)[0]

    # Calculate the correlation map over the whole set
    corr = np.zeros((image_all[0].shape[0], image_all[0].shape[1], image_all[0].shape[2]))
    p = np.zeros((image_all[0].shape[0], image_all[0].shape[1], image_all[0].shape[2]))

    for j in range(image_all[0].shape[0]):
        for k in range(image_all[0].shape[1]):
            for l in range(image_all[0].shape[2]):
                x = image_all[idx_train, j, k, l]
                y = decoding_measure[idx_train]
                # Mask out NaNs
                mask = ~np.isnan(x) & ~np.isnan(y)
                corr[j, k, l], p[j, k, l] = spearmanr(x[mask], y[mask])

    # Save the correlation map for further analysis
    folder_path = f"../../Maps/Correlation/{model_name}/flip_{flip_mode}/{mode}/across_subjects_all"
    os.makedirs(folder_path, exist_ok=True)
    savemat(f"{folder_path}/without_{i}_func_seed_AvgR_Fz.mat", {"corr": corr, "p": p})

    # Save as nifti file
    path = f"../../Maps/Functional_connectivity/Subject_1_bipolar_1_func_seed_AvgR_Fz.nii"  # Assuming a path for NIfTI
    nii = nib.load(path)
    nii_new = nib.Nifti1Image(corr, nii.affine, nii.header)
    nib.save(nii_new, f"{folder_path}/without_{i}_func_seed_AvgR_Fz.nii")


# Use joblib.Parallel to parallelize the computation
Parallel(n_jobs=-1)(delayed(compute_corr_maps)(i, image_all, decoding_measure, df_decoding, model_name) for i in np.hstack((0, subjects)))
