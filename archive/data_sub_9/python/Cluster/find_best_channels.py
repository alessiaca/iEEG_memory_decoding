# Across patient decoding using the optimal channel based on the functional connectivity

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


def find_best_channels(df, subject, decoding_mode, subject_mode, flip_mode, overlap_thresh, mask):
    """Given a set of parameters, compute the similarity of each channel and patient to the r-map calculated on the remaining patients"""

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
    df = df.reset_index()
    image_all = []
    for _, row in df.iterrows():
        subject_tmp = int(row["subject"])
        channel = int(row["channel"])

        flip_options = ["true", "false"] if flip_mode == "true" else ["false"]
        for flipped in flip_options:
            #path = f"functional_connectivity_maps/Subject_{subject}_None_{channel}_flipped_{flipped}_func_seed_AvgR_Fz.nii"
            path = f"../../../Maps/Functional_connectivity/Subject_{subject_tmp}_none_{channel}_flipped_{flipped}_func_seed_AvgR_Fz.nii"
            if os.path.exists(path):
                break
        image = nib.load(path).get_fdata()
        image_all.append(image)

    image_all = np.array(image_all)

    # Leave one patient out
    idx_train = np.where(df["subject"] != subject)[0]
    
    # Compute spearman correlation between decoding accuracy and functional connectivity for each voxel _______________________________________________________

    # Reshape data
    X = image_all[idx_train].reshape(len(idx_train), -1) 
    Y = decoding_accuracy[idx_train]

    # Initalize r-map with NaNs
    r_vals = np.full(X.shape[1], np.nan)

    # Pre-rank decoding accuracies
    y_ranked = rankdata(Y)
    y_z = (y_ranked - y_ranked.mean()) / y_ranked.std()

    for v in range(X.shape[1]):
        x = X[:, v]
        valid = ~np.isnan(x)
        if valid.sum() < 2 or np.ptp(x[valid]) == 0:
            continue
        x_rank = rankdata(x[valid])
        x_z = (x_rank - x_rank.mean()) / x_rank.std()
        y_z = (y_ranked[valid] - y_ranked[valid].mean()) / y_ranked[valid].std()

        r_vals[v] = np.dot(x_z, y_z) / (len(x_z) - 1)

    r_map = r_vals.reshape(image_all.shape[1:])

    # Calculate the similiarity between the r-map and the functional connectivity of each channel of the left-out patient ___________________________________
    results_sub = []
    for idx, row in df.iterrows():
        subject_tmp = row["subject"]
        channel = row["channel"]
        image_flat = image_all[idx].flatten()
        r_flat = r_map.flatten()
        if mask:
            valid = (~np.isnan(image_flat)) & (~np.isnan(r_flat)) & (mask_image.flatten() > 0.1)
        else:
            valid = (~np.isnan(image_flat)) & (~np.isnan(r_flat))
        similarity, _ = spearmanr(image_flat[valid], r_flat[valid])
        results_sub.append([subject, channel, subject_tmp, similarity])

    # Create a DataFrame from the collected results
    results_df = pd.DataFrame(results_sub, columns=['subject', 'channel', 'subject_tmp', 'similarity'])

    # Get the channels with the highest similarity for each subject
    best_channels_df = results_df.loc[results_df.groupby('subject_tmp')['similarity'].idxmax()].reset_index(drop=True)
    
    # Write to database
    with sqlite3.connect(f"best_channels/{subject}_{decoding_mode}.db", timeout=30) as conn:
        best_channels_df.to_sql("selected_data", conn, if_exists="append", index=False)

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
    decoding_mode = "5_3_1"
    subject_mode = np.array([2, 3, 6, 7, 8, 9])
    flip_mode = "false"
    overlap_thres = None
    mask = None
    combinations = subject_mode

    # Run
    if cluster:
        input1 = int(sys.argv[1])-1 
        find_best_channels(df, combinations[input1], decoding_mode, subject_mode, flip_mode, overlap_thres, mask)
    else:
        Parallel(n_jobs=2)(delayed(find_best_channels)(df, subject, decoding_mode, subject_mode, flip_mode, overlap_thres, mask) for subject in subject_mode)
