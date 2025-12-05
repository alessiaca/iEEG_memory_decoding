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

# Set parameters
referencing = "none"
Hz = 5
length = 1000
n_runs = 200
model_name = f"{referencing}_Hz_{Hz}_length_{length}_runs_{n_runs}"

# Options to loop over
subject_mode = np.array([2, 3, 5, 6, 7, 8, 9])#  [[3, 6, 7, 8, 9], np.arange(1, 10), np.arange(2, 10)]
flip_mode = "true"
overlap_thresh = 0.25

# Load input data
decoding_df = pd.read_csv(f"../../Decoding_results/{model_name}.csv")

# Filter > 50 % decoding performance 
decoding_df = decoding_df[decoding_df["accuracy"] > 0.5]
# Filter subjects of interest
decoding_df = decoding_df[decoding_df["subject"].isin(subject_mode)]
# Filter channels based on grey matter overlap
if overlap_thresh:
    decoding_df= decoding_df[decoding_df["percent_overlap"] > overlap_thresh]

# Load the intersecting tracts (indices)
tracts_df = pd.read_csv("../../Fiber_filtering/tracts_overlap.csv")

tracts_df = tracts_df.merge(
            decoding_df[["subject", "channel"]].drop_duplicates(),
            on=["subject", "channel"],
            how="inner"
        )

# Get all tracts to loop over
if flip_mode == "false":
    tracts_df = tracts_df[tracts_df["flipped"] == False]
else:
    flipped = tracts_df[tracts_df["flipped"] == True]
    false_only = tracts_df[
        (tracts_df["flipped"] == False) &
        (~tracts_df[["subject", "channel"]].apply(tuple, axis=1).isin(
            flipped[["subject", "channel"]].apply(tuple, axis=1)
        ))
    ]
    tracts_df = pd.concat([flipped, false_only], ignore_index=True)

# Precompute: for each tract, the set of (subject, channel)
tract_to_pairs = tracts_df.groupby("tract_index")[["subject", "channel"]].apply(
    lambda df: set(map(tuple, df.to_numpy()))
).to_dict()

# Precompute the full set of all pairs
all_pairs = set(map(tuple, tracts_df[["subject", "channel"]].drop_duplicates().to_numpy()))

decoding_lookup = decoding_df.set_index(["subject", "channel"])["accuracy"]

# Calculate the discriminative tracts on the remaining patients
tracts_unique = tracts_df["tract_index"].unique()
t_p_stats = []

for tract_idx in tqdm(tracts_unique, desc="Discriminative tracts"):

    connected_set = tract_to_pairs.get(tract_idx, set())
    not_connected_set = all_pairs - connected_set

    # Filter out the left-out subject from both sets
    connected_acc = [decoding_lookup.get(key) for key in connected_set if key in decoding_lookup]
    not_connected_acc = [decoding_lookup.get(key) for key in not_connected_set if key in decoding_lookup]

    # Calculate the t-statistic and p-value
    #n_thres = len(decoding_df) * 0.3
    ratio = len(connected_acc)/len(not_connected_acc)
    if len(connected_acc) > 1 and len(not_connected_acc) > 1:
        T, p = ttest_ind(connected_acc, not_connected_acc)
        t_p_stats.append([tract_idx, T, p, ratio])
    else:
        t_p_stats.append([tract_idx, 0, 1, ratio])

t_p_stats_df = pd.DataFrame(t_p_stats, columns=['tract_index', 't_statistic', 'p_value', "ratio"])
t_p_stats_df.to_csv(f"../../Fiber_filtering//t_p_fibers_{flip_mode}_{overlap_thresh}_{np.mean(subject_mode)}.csv")