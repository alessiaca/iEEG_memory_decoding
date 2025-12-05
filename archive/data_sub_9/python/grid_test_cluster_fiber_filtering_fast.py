import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import sqlite3
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm


def process_combination(subject_mode, flip_mode, overlap_thresh, decoding_df, tracts_df):
    # Filter decoding > 0.5 and subjects of interest
    decoding_df = decoding_df[decoding_df["accuracy"] > 0.5]
    decoding_df = decoding_df[decoding_df["subject"].isin(subject_mode)]

    # Filter channels based on grey matter overlap (using mask)
    if overlap_thresh is not None:
        mask = overlap_df["percent_overlap"] > overlap_thresh
        decoding_df = decoding_df.loc[mask.values]

    flipped_bool = (flip_mode == "true")
    tracts_df = tracts_df[tracts_df["flipped"] == flipped_bool]

    # Convert subject and channel to categorical for speed
    for df in [decoding_df, tracts_df]:
        df["subject"] = df["subject"].astype("category")
        df["channel"] = df["channel"].astype("category")

    # Prepare arrays and mappings
    subject_codes = decoding_df["subject"].cat.codes.values
    channel_codes = decoding_df["channel"].cat.codes.values

    # Unique pairs from tracts_df
    unique_pairs_df = tracts_df[["subject", "channel"]].drop_duplicates()
    unique_pairs_df["subj_code"] = unique_pairs_df["subject"].cat.codes
    unique_pairs_df["chan_code"] = unique_pairs_df["channel"].cat.codes

    all_pairs = list(zip(unique_pairs_df["subj_code"], unique_pairs_df["chan_code"]))
    pair_to_idx = {pair: i for i, pair in enumerate(all_pairs)}

    # decoding accuracy aligned to (subject, channel) pairs, fill nan if missing
    decoding_map = decoding_df.set_index(["subject", "channel"])["accuracy"]
    acc_array = np.full(len(all_pairs), np.nan)
    for pair, idx in pair_to_idx.items():
        subj_cat = decoding_df["subject"].cat.categories[pair[0]]
        chan_cat = decoding_df["channel"].cat.categories[pair[1]]
        acc_array[idx] = decoding_map.get((subj_cat, chan_cat), np.nan)

    # Group tracts by tract_index: map tract -> set of indices in all_pairs
    tract_groups = {}
    for tract_idx, group in tracts_df.groupby("tract_index"):
        # get pairs as categorical codes and map to indices
        pairs = list(zip(group["subject"].cat.codes, group["channel"].cat.codes))
        indices = [pair_to_idx[p] for p in pairs if p in pair_to_idx]
        mask = np.zeros(len(all_pairs), dtype=bool)
        mask[indices] = True
        tract_groups[tract_idx] = mask

    tracts_unique = list(tract_groups.keys())

    # Precompute discriminative tracts stats
    t_p_stats = []
    for tract_idx in tqdm(tracts_unique, desc="Discriminative tracts"):
        connected_mask = tract_groups[tract_idx]
        connected_perf = acc_array[connected_mask]
        not_connected_perf = acc_array[~connected_mask]

        connected_perf = connected_perf[~np.isnan(connected_perf)]
        not_connected_perf = not_connected_perf[~np.isnan(not_connected_perf)]

        if len(connected_perf) > 10 and len(not_connected_perf) > 10:
            T, p = ttest_ind(connected_perf, not_connected_perf)
        else:
            T, p = 0, 1

        t_p_stats.append([tract_idx, T, p])
    t_p_stats_df = pd.DataFrame(t_p_stats, columns=['tract_index', 't_statistic', 'p_value'])

    # Loop over subjects (leave-one-out)
    results_sub = []
    for subject in subject_mode:
        idx_test = decoding_df.index[decoding_df["subject"] == subject]

        # For each channel of the left-out patient
        for idx in idx_test:
            channel = decoding_df.at[idx, "channel"]
            accuracy = decoding_df.at[idx, "accuracy"]

            # Find connected tracts for this subject & channel
            connected_tracts = tracts_df[
                (tracts_df["subject"] == subject) & (tracts_df["channel"] == channel)
            ]["tract_index"].unique()

            if len(connected_tracts) > 0:
                t_vals = t_p_stats_df[t_p_stats_df["tract_index"].isin(connected_tracts)]["t_statistic"].values
                fiber_score = np.mean(t_vals)
            else:
                fiber_score = np.nan

            results_sub.append([np.mean(subject_mode), flip_mode, overlap_thresh, subject, channel, fiber_score, accuracy])

    results_df = pd.DataFrame(results_sub, columns=['subject_mode', 'flip_mode', 'overlap', 'mask', 'subject', 'channel', 'similarity', 'accuracy'])

    # Write results to SQLite in one batch
    conn = sqlite3.connect("results_fiber_filtering.db")
    results_df.to_sql("selected_data", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    referencing = "none"
    Hz = 5
    length = 1000
    n_runs = 200
    model_name = f"{referencing}_Hz_{Hz}_length_{length}_runs_{n_runs}"

    subject_modes = [[3, 6, 7, 8, 9], np.arange(1, 10), np.arange(2, 10)]
    flip_modes = ["false", "true"]
    overlap_thresholds = [0.25, None, 0.0]

    decoding_df = pd.read_csv(f"../../Decoding_results/{model_name}.csv")
    overlap_df = pd.read_csv("../../Processed_data/metadata/grey_matter_overlap.csv")
    tracts_df = pd.read_csv("../../Maps/tracts_overlap.csv")

    combinations = list(product(subject_modes, flip_modes, overlap_thresholds))

    # Parallel with n_jobs>1 if you want faster; 1 for debugging
    Parallel(n_jobs=1)(
        delayed(process_combination)(subject_mode, flip_mode, overlap_thresh, decoding_df, overlap_df, tracts_df)
        for subject_mode, flip_mode, overlap_thresh in combinations
    )
