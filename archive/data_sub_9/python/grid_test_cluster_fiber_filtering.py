import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr, ttest_ind
import sqlite3
from scipy.io import loadmat
from itertools import product
from joblib import Parallel, delayed
import os
from tqdm import tqdm
import sys

def process_combination(subject_mode, flip_mode, overlap_thresh, decoding_df, tracts_df):

    # Filter > 50 % decoding performance 
    decoding_df = decoding_df[decoding_df["accuracy"] > 0.5]
    # Filter subjects of interest
    decoding_df = decoding_df[decoding_df["subject"].isin(subject_mode)]
    # Filter channels based on grey matter overlap
    if overlap_thresh:
        decoding_df = decoding_df[decoding_df["percent_overlap"] > overlap_thresh]
    tracts_df = tracts_df.merge(
                decoding_df[["subject", "channel"]].drop_duplicates(),
                on=["subject", "channel"],
                how="inner"
            )

    # Get all tracts to loop over
    flipped_bool = False if flip_mode == "false" else True
    tracts_df = tracts_df[tracts_df["flipped"] == flipped_bool]

    # Precompute: for each tract, the set of (subject, channel)
    tract_to_pairs = tracts_df.groupby("tract_index")[["subject", "channel"]].apply(
        lambda df: set(map(tuple, df.to_numpy()))
    ).to_dict()

    # Precompute the full set of all pairs
    all_pairs = set(map(tuple, tracts_df[["subject", "channel"]].drop_duplicates().to_numpy()))

    decoding_lookup_all = decoding_df.set_index(["subject", "channel"])["accuracy"]

    # Loop over subject
    for subject in subject_mode:

        decoding_excl = decoding_df[decoding_df["subject"] != subject]
        decoding_lookup = decoding_excl.set_index(["subject", "channel"])["accuracy"]

        # Calculate the discriminative tracts on the remaining patients
        tracts_unique = tracts_df[tracts_df["subject"] == subject]["tract_index"].unique()
        t_p_stats = []

        for tract_idx in tqdm(tracts_unique, desc="Discriminative tracts"):

            connected_set = tract_to_pairs.get(tract_idx, set())
            not_connected_set = all_pairs - connected_set

            # Filter out the left-out subject from both sets
            connected_acc = [decoding_lookup.get(key) for key in connected_set if key in decoding_lookup]
            not_connected_acc = [decoding_lookup.get(key) for key in not_connected_set if key in decoding_lookup]

            # Calculate the t-statistic and p-value
            if len(connected_acc) > 5 and len(not_connected_acc) > 5:
                T, p = ttest_ind(connected_acc, not_connected_acc)
                t_p_stats.append([tract_idx, T, p])
            else:
                t_p_stats.append([tract_idx, 0, 1])

        t_p_stats_df = pd.DataFrame(t_p_stats, columns=['tract_index', 't_statistic', 'p_value'])

        # Pre-filter tracts for the left-out subject
        tracts_sub = tracts_df[tracts_df["subject"] == subject]

        # Pre-index t-statistics for fast lookup
        t_stat_map = t_p_stats_df.set_index("tract_index")["t_statistic"].to_dict()

        results_sub = []
        for row in tracts_sub[["subject", "channel"]].drop_duplicates().itertuples(index=False):
            channel = row.channel

            # Get connected tracts
            connected_tracts = tracts_sub[tracts_sub["channel"] == channel]["tract_index"].unique()

            # Average t-statistic if available
            if connected_tracts.size > 0:
                fiber_scores = [t_stat_map.get(t, 0) for t in connected_tracts]
                fiber_score = np.mean(fiber_scores)
            else:
                fiber_score = np.nan  # or 0, depending on your logic

            # Collect results
            results_sub.append([
                np.mean(subject_mode),  # if subject_mode is a list of int/floats
                flip_mode,
                overlap_thresh,
                subject,
                channel,
                fiber_score,
                decoding_lookup_all.get((subject, channel), np.nan) 
            ])

        # Create a DataFrame from the collected results
        results_df = pd.DataFrame(results_sub, columns=['subject_mode', 'flip_mode', 'overlap', 'subject', 'channel', 'similarity', 'accuracy'])

        # Connect to SQLite database once
        conn = sqlite3.connect(f"results_fiber_filtering.db")
        
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

    # Options to loop over
    subject_modes = [[3, 6, 7, 8, 9], np.arange(1, 10), np.arange(2, 10)]
    flip_modes = ["false", "true"]
    overlap_thresholds = [0.25, None, 0.0]

    # Load data
    decoding_df = pd.read_csv(f"../../Decoding_results/{model_name}.csv")

    # Load the intersecting tracts (indices)
    tracts_df = pd.read_csv("../../Maps/tracts_overlap.csv")

    # Prepare job list
    combinations = list(product(subject_modes, flip_modes, overlap_thresholds))

    # Run in parallel
    Parallel(n_jobs=1)(
        delayed(process_combination)(subject, flip_mode, overlap,
                                     decoding_df.copy(), tracts_df.copy())
        for subject, flip_mode, overlap in combinations
    )
