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


def calculate_overlap(subject, channel, flipped):

    # Load the electrode ROI nii files
    path = f"../../Electrode_nifti/Subject_{subject}_{referencing}_{channel}_flipped_{flipped}.nii"
    mask_img = nib.load(path)
    data = mask_img.get_fdata() > 0  # binary mask
    affine = mask_img.affine  #

    # Load the tracts (HCP 1000)
    tractogram = nib.streamlines.load("../../Atlas/HCP_1000_tracts.trk")
    streamlines = tractogram.streamlines

    sft = StatefulTractogram(tractogram.streamlines, mask_img, Space.RASMM)

    # --- Get intersecting streamlines using `target()` ---
    intersecting_generator = target(sft.streamlines, affine, data, include=True)

    # --- Convert generator to list of intersecting streamlines ---
    intersecting_idx = list(intersecting_generator)

    # Prepare results for saving
    results = [
        [subject, channel, flipped, idx]
        for idx in intersecting_idx
    ]

    # Save to database
    results_df = pd.DataFrame(results, columns=['subject', 'channel', 'flipped', 'tract_index'])

    # Connect to SQLite database 
    conn = sqlite3.connect(f"tract_overlap.db")
    
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
    df_decoding = pd.read_csv(f"../../Decoding_results/{model_name}.csv")
    flipped = "false"

    # Run in parallel
    Parallel(n_jobs=1)(
        delayed(calculate_overlap)(subject, channel, flipped)
        for subject, channel in df_decoding[['subject', 'channel']].itertuples(index=False)
    )
