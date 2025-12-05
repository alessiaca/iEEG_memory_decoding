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


def compute_r_map(df, decoding_mode, subject_mode, flip_mode, overlap_thresh, mask):
    """Given a set of parameters, compute the similarity of each channel and patient to the r-map calculated on the remaining patients"""

    # Get the decoding accuracy based on the decoding mode
    if decoding_mode == "max":
        # Get the highest accuracy for each subject/channel
        idx = df.groupby(["subject", "channel"])["accuracy"].idxmax()
        df = df.loc[idx].reset_index(drop=True)
    else:
        # Get a specific combination of decoding parameters
        df = df[df["decoding_params"] == decoding_mode]

    # Get the mask image
    if mask:
        mask_image = nib.load(f"WM_tmp/{mask}.nii").get_fdata()

    # Filter > 50 % decoding performance and subjects not of interest
    df = df[(df["accuracy"] > 0.5) & (df["subject"].isin(subject_mode))]
    # Filter channels based on grey matter overlap
    if overlap_thresh:
        df = df[df["percent_overlap"] > overlap_thresh]

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
    nib.save(nii_new, f"r_p_maps/r_{decoding_mode}_{subject_mode.mean()}_{flip_mode}_{overlap_thresh}_{mask}.nii")
    nii_new = nib.Nifti1Image(p, nii.affine, nii.header)
    nib.save(nii_new, f"r_p_maps/p_{decoding_mode}_{subject_mode.mean()}_{flip_mode}_{overlap_thresh}_{mask}.nii")
    corr[p > 0.05] = np.NaN
    nii_new = nib.Nifti1Image(corr, nii.affine, nii.header)
    nib.save(nii_new, f"r_p_maps/r_p_masked_{decoding_mode}_{subject_mode.mean()}_{flip_mode}_{overlap_thresh}_{mask}.nii")
    p[p > 0.05] = np.NaN
    p[p <= 0.05] = 1
    nii_new = nib.Nifti1Image(p, nii.affine, nii.header)
    nib.save(nii_new, f"r_p_maps/p_masked_{decoding_mode}_{subject_mode.mean()}_{flip_mode}_{overlap_thresh}_{mask}.nii")



if __name__ == "__main__":

    cluster = False

    # Load the overlap with the grey matter
    df_overlap = pd.read_csv("WM_tmp/grey_matter_overlap.csv")

    # Load the decoding accuracies from folder 
    db_files = glob.glob(os.path.join('decoding_results/', '*.db'))
    dfs = []
    for db in db_files:
        with sqlite3.connect(db) as conn:
            dfs.append(pd.read_sql(f"SELECT * FROM selected_data", conn))
    df = pd.concat(dfs, ignore_index=True)
    
    # Keep only non-shuffled results and add a new column summarizing the decoding parameters
    df = df[df["shuffled"] == 0]
    df.loc[:, "decoding_params"] = df[["seconds", "components", "sampling_hz"]].astype(str).agg("_".join, axis=1)
    
    # Calculate the mean accuracy across all runs 
    df = df.groupby(["subject", "channel", "decoding_params"])["accuracy"].mean().reset_index()

    # Merge with overlap
    df = df.merge(df_overlap[["subject", "channel", "percent_overlap"]], on=["subject", "channel"], how="left")

    # Grid search parmeter to parallelize over
    decoding_mode = "3_3_2"
    subject_mode = np.array([2, 3, 6, 7, 8, 9])
    flip_mode = "false"
    overlap_threshold = None
    mask = None

    # Run
    if cluster:
        compute_r_map(df, decoding_mode, subject_mode, flip_mode, overlap_threshold, mask)
    else:
        compute_r_map(df, decoding_mode, subject_mode, flip_mode, overlap_threshold, mask)
