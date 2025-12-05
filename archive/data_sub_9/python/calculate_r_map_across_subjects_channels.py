# R_map calculation across subjects and channels 

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr
from scipy.io import savemat
import os
from joblib import Parallel, delayed

# Define parameters
referencing = "bip"
Hz = 10
length = 1000
n_runs = 100
model_name = f"{referencing}_Hz_{Hz}_length_{length}_runs_{n_runs}"
measure = "accuracy"
subjects = np.arange(1, 10)
remove_below_chance = False
flip_mode = "false"

# Load the decoding results
df_decoding = pd.read_csv(f"../../Decoding_results/{model_name}.csv")

# Filter the channels which have above-chance level decoding for all runs (with random seeds)
if remove_below_chance:
    df_decoding = df_decoding[df_decoding["above_chance"]]

# Load the functional connectivity maps for each subject/channel combination
image_all = []
df_decoding_new = df_decoding.copy()
for i, row in df_decoding_new.iterrows():
    subject = int(row["subject"])
    channel = int(row["channel"])

    flip_options = ["true", "false"] if flip_mode == "all" else [flip_mode]
    path = None
    try:
        for flipped in flip_options:
            path = f"../../Maps/Functional_connectivity/Subject_{subject}_bipolar_{channel}_flipped_{flipped}_func_seed_AvgR_Fz.nii"
            if os.path.exists(path):
                break
        image = nib.load(path).get_fdata()
        image_all.append(image)
    except:
        df_decoding = df_decoding.drop(i, axis=0)

image_all = np.array(image_all)

# Extract the measure
decoding_measure = df_decoding[measure].to_numpy()

# Function to compute correlation maps for a given (subject, channel) combination
def compute_corr_maps(i, image_all, decoding_measure, df_decoding, model_name):

    # Exclude the data for this (subject, channel) pair
    if i == len(df_decoding):
        idx_train = np.arange(len(df_decoding))
        subject = 0
        channel = 0
    else:
        idx_train = np.setdiff1d(np.arange(len(df_decoding)), i)  # All except current index i
        # Get the (subject, channel) pair to exclude
        subject = df_decoding.iloc[i]["subject"]
        channel = df_decoding.iloc[i]["channel"]

    # Calculate the correlation map over the whole set
    corr = np.zeros((image_all[0].shape[0], image_all[0].shape[1], image_all[0].shape[2]))
    p = np.zeros((image_all[0].shape[0], image_all[0].shape[1], image_all[0].shape[2]))

    for j in range(image_all[0].shape[0]):
        for k in range(image_all[0].shape[1]):
            for l in range(image_all[0].shape[2]):
                corr[j, k, l], p[j, k, l] = spearmanr(image_all[idx_train, j, k, l], decoding_measure[idx_train])

    # Save the correlation map for further analysis
    folder_path = f"../../Maps/Correlation/{model_name}/flip_{flip_mode}/across_subjects_channels/"
    os.makedirs(folder_path, exist_ok=True)
    savemat(
        f"{folder_path}/without_subject_{subject}_channel_{channel}_func_seed_AvgR_Fz.mat",
        {"corr": corr, "p": p})

    # Save as nifti file
    path = f"../../Maps/Functional_connectivity/Subject_1_bipolar_1_func_seed_AvgR_Fz.nii"  # Assuming a path for NIfTI
    nii = nib.load(path)
    nii_new = nib.Nifti1Image(corr, nii.affine, nii.header)
    nib.save(nii_new,
             f"{folder_path}/without_subject_{subject}_channel_{channel}_func_seed_AvgR_Fz.nii")


# Use joblib.Parallel to parallelize the computation
Parallel(n_jobs=-1)(
    delayed(compute_corr_maps)(i, image_all, decoding_measure, df_decoding, model_name) for i in
    range(len(df_decoding)+1))
