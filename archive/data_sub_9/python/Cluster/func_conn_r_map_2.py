# Calculate the r-map for a specific parameter combination for visualization purposes

import numpy as np
import glob
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr, pearsonr, rankdata
import sqlite3
from itertools import product
from joblib import Parallel, delayed
import os
import time
import sys


def compute_p_value(group):
    true = group[group["shuffled"] == 0]["accuracy"]
    perm = group[group["shuffled"] == 1]["accuracy"]

    perm = np.array(perm)          # shape (n_perm,)
    true = float(true)            # observed statistic

    n_perm = len(perm)
    count = np.sum(perm >= true)  # one-sided: count permutations >= observed
    p_value = (count + 1) / (n_perm + 1)
    return p_value


def compute_r_map(df, flip_mode, p_thresh):
    """Given a set of parameters, compute the similarity of each channel and patient to the r-map calculated on the remaining patients"""

    # Filter > 50 % decoding performance and p value below threshold
    df = df[df["p_value"] < p_thresh]

    # Flip the decoding accuracies 
    #df[df.accuracy] < 0.5 = 0
    #df.loc[df.accuracy < 0.5, "accuracy"] = 1 - df.loc[df.accuracy < 0.5, "accuracy"]

    decoding_accuracy = df["accuracy"].to_numpy()

    # Load functional connectivity maps
    image_all = []
    for _, row in df.iterrows():
        subject = int(row["subject"])
        channel = int(row["channel"])

        flip_options = ["true", "false"] if flip_mode == "true" else ["false"]
        for flipped in flip_options:
            #path = f"functional_connectivity_maps/Subject_{subject}_None_{channel}_flipped_{flipped}_func_seed_AvgR_Fz.nii"
            path = f"../../../Maps/Functional_connectivity/Subject_{subject}_none_{channel}_flipped_{flipped}_func_seed_AvgR_Fz.nii"
            if os.path.exists(path):
                break
        image = nib.load(path).get_fdata()
        image_all.append(image)

    image_all = np.array(image_all)
        
    # Compute spearman correlation between decoding accuracy and functional connectivity for each voxel
    corr = np.full((image.shape[0], image.shape[1], image.shape[2]), np.NaN)
    p = np.full((image.shape[0], image.shape[1], image.shape[2]), np.NaN)
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            for l in range(image.shape[2]):
                valid = ~np.isnan(image_all[:, j, k, l]) & ~np.isnan(decoding_accuracy)
                corr[j, k, l], p[j, k, l] = spearmanr(image_all[valid, j, k, l], decoding_accuracy[valid])

    # Save as nifti files
    nii = nib.load(path)
    nii_new = nib.Nifti1Image(corr, nii.affine, nii.header)
    nib.save(nii_new, f"r_p_maps/r_{flip_mode}_{p_thresh}.nii")
    nii_new = nib.Nifti1Image(p, nii.affine, nii.header)
    nib.save(nii_new, f"r_p_maps/p_{flip_mode}_{p_thresh}.nii")
    corr[p > 0.05] = np.NaN
    nii_new = nib.Nifti1Image(corr, nii.affine, nii.header)
    nib.save(nii_new, f"r_p_maps/r_p_masked_{flip_mode}_{p_thresh}.nii")
    p[p > 0.05] = np.NaN
    p[p <= 0.05] = 1
    nii_new = nib.Nifti1Image(p, nii.affine, nii.header)
    nib.save(nii_new, f"r_p_maps/p_masked_{flip_mode}_{p_thresh}.nii")



if __name__ == "__main__":

    # Load the decoding accuracies
    folder_path = "decoding_results_2 (24)"

    # Get list of all CSV files
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # Load and concatenate all CSVs
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    
    # Select only seconds of interest (maintenance)
    df = df[df["sampling_hz"] == 3]

    # Compute thep values
    df_p = (df.groupby(["subject", "channel"]).apply(compute_p_value).reset_index(name="p_value"))

    # Merge
    df = df[df["shuffled"] == 0]
    df_merged = pd.merge(df, df_p, on=['subject', 'channel'], how='left')

    # Filter out subjects of interest
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    df_merged = df_merged[df_merged["subject"].isin(subjects)]

    flip_mode = "false"
    p_threshold = 0.5

    # Run
    compute_r_map(df_merged, flip_mode, p_threshold)
