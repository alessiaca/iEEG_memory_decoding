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


def compute_r_map(df):
    """Given a set of parameters, compute the similarity of each channel and patient to the r-map calculated on the remaining patients"""

    decoding_accuracy = df["accuracy"].to_numpy()

    # Load functional connectivity maps
    image_all = []
    for _, row in df.iterrows():
        subject = int(row["subject"])
        channel = int(row["channel"])
        path = f"../../../Maps/Functional_connectivity/Subject_{subject}_none_{channel+1}_flipped_false_func_seed_AvgR_Fz.nii"
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
    nib.save(nii_new, f"r_p_maps/r_{name}.nii")
    nii_new = nib.Nifti1Image(p, nii.affine, nii.header)
    nib.save(nii_new, f"r_p_maps/p_{name}.nii")
    corr[p > 0.05] = np.NaN
    nii_new = nib.Nifti1Image(corr, nii.affine, nii.header)
    nib.save(nii_new, f"r_p_maps/r_p_masked_{name}.nii")
    p[p > 0.05] = np.NaN
    p[p <= 0.05] = 1
    nii_new = nib.Nifti1Image(p, nii.affine, nii.header)
    nib.save(nii_new, f"r_p_maps/p_masked_{name}.nii")



if __name__ == "__main__":

    cluster = False

    df = pd.read_csv("WM_load_decoding_performance.csv")
    subjects = [2,3,6,7,8,9]
    
    # Keep only non-shuffled results and add a new column summarizing the decoding parameters
    df = df[df["shuffled"] == 0]
    df = df[df["subject"].isin(subjects)]
    # Calculate the mean accuracy across all runs 
    df = df.groupby(["subject", "channel", "period"])["accuracy"].mean().reset_index()
    
    for period in df.period.unique():
        compute_r_map(df[df.period==period], period)