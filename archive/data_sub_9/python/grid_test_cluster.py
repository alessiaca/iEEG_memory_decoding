import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr, rankdata
import sqlite3
from itertools import product
from joblib import Parallel, delayed
import os


def process_combination(subject_mode, flip_mode, overlap_thresh, mask, decoding_df):

    # Get the mask image
    if mask:
        mask_image = nib.load(f"{mask}.nii").get_fdata()

    # Filter > 50 % decoding performance 
    decoding_df = decoding_df[decoding_df["accuracy"] > 0.5]
    # Filter subjects of interest
    decoding_df = decoding_df[decoding_df["subject"].isin(subject_mode)]
    # Filter channels based on grey matter overlap
    if overlap_thresh:
        decoding_df = decoding_df[decoding_df["percent_overlap"] > overlap_thresh]

    decoding_array = decoding_df["accuracy"].to_numpy()

    # Load functional connectivity maps
    image_all = []
    for i, row in decoding_df.iterrows():
        subject = int(row["subject"])
        channel = int(row["channel"])

        flip_options = ["true", "false"] if flip_mode == "true" else ["false"]
        for flipped in flip_options:
            path = f"../../Maps/Functional_connectivity/Subject_{subject}_{referencing}_{channel}_flipped_{flipped}_func_seed_AvgR_Fz.nii"
            if os.path.exists(path):
                break
        image = nib.load(path).get_fdata()
        image_all.append(image)

    image_all = np.array(image_all)

    # Loop over subject
    for subject in subject_mode:

        # Leave one patient out
        idx_train = np.where(decoding_df["subject"] != subject)[0]
        idx_test = np.where(decoding_df["subject"] == subject)[0]

        # Reshape and rank data
        X = image_all[idx_train].reshape(len(idx_train), -1)  # shape (n_train, n_voxels)
        Y = decoding_array[idx_train]
        
        # Drop voxels with NaNs or constant timecourses
        valid_voxel_mask = ~np.isnan(X).any(axis=0) & (X.ptp(axis=0) != 0)  # remove constant columns
        
        # Rank-transform (Spearman = Pearson on ranks)
        X_ranked = np.apply_along_axis(rankdata, 0, X[:, valid_voxel_mask])
        Y_ranked = rankdata(Y)
        
        # Z-score the ranks
        X_z = (X_ranked - X_ranked.mean(axis=0)) / X_ranked.std(axis=0)
        Y_z = (Y_ranked - Y_ranked.mean()) / Y_ranked.std()
        
        # Compute correlations via dot product
        r_vals = X_z.T @ Y_z / (len(Y) - 1)
        
        # Reconstruct full r_map (with NaNs in invalid voxels)
        r_full = np.full(X.shape[1], np.nan)
        r_full[valid_voxel_mask] = r_vals
        r_map = r_full.reshape(image_all.shape[1:])

        # Calculate the similiarity between the optimal r-map and the functional connectivity of each channel of the left-out patient
        results_sub = []
        for idx, channel, accuracy in zip(idx_test, decoding_df["channel"].iloc[idx_test], decoding_df["accuracy"].iloc[idx_test]):
            image = image_all[idx, :, :, :]
            if mask:
                valid = ~np.isnan(image.flatten()) & ~np.isnan(r_map.flatten()) & (mask_image.flatten() > 0.1)
            else:
                valid = ~np.isnan(image.flatten()) & ~np.isnan(r_map.flatten())
            similarity, _ = spearmanr(image.flatten()[valid], r_map.flatten()[valid])
            results_sub.append([subject_mode.mean(), flip_mode, overlap_thresh, mask, subject, channel, similarity, accuracy])

        # Create a DataFrame from the collected results
        results_df = pd.DataFrame(results_sub, columns=['subject_mode', 'flip_mode', 'overlap', 'mask', 'subject', 'channel', 'similarity', 'accuracy'])

        # Connect to SQLite database once
        conn = sqlite3.connect(f"results.db")
        
        # Append the entire dataframe to the database
        results_df.to_sql("selected_data", conn, if_exists="append", index=False)

        # Commit and close the connection
        conn.commit()
        conn.close()

if __name__ == "__main__":

    # Set parameters
    referencing = "none"
    Hz = 5
    length = 1000
    n_runs = 200
    model_name = f"{referencing}_Hz_{Hz}_length_{length}_runs_{n_runs}"
    """decoding_path = f"WM_tmp/{model_name}.csv"
    overlap_path = f"WM_tmp/grey_matter_overlap.csv"
    map_folder = f"functional_connectivity_maps"""

    # Options to loop over
    subject_modes = [[3, 6, 7, 8, 9], np.arange(1, 10), np.arange(2, 10)]
    flip_modes = ["true", "false"]
    overlap_thresholds = [None, 0.25, None, 0.0]
    masks = ["memory_association-test_z_FDR_0.01", None, "rFornix", "rnucleus_accumbens", "rc1mask"]

    # Load input data
    df_decoding = pd.read_csv(f"../../Decoding_results/{model_name}.csv")
    df_overlap = pd.read_csv("../../Processed_data/metadata/grey_matter_overlap.csv")

    # Prepare job list
    combinations = list(product(subject_modes, flip_modes, overlap_thresholds, masks))

    # Run in parallel
    Parallel(n_jobs=1)(
        delayed(process_combination)(subject, flip_mode, overlap, mask,
                                     df_decoding.copy())
        for subject, flip_mode, overlap, mask in combinations
    )
