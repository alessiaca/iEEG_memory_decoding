# Calculate the statistics of the WM decoding accuracy (against a shuffled version)

import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns
import glob
from scipy.stats import wilcoxon
matplotlib.use("TkAgg")


def plot_WM_decoding(combination):

    # Load the decoding accuracies from folder 
    db_files = glob.glob(os.path.join('decoding_results/', '*.db'))
    dfs = []
    for db in db_files:
        with sqlite3.connect(db) as conn:
            dfs.append(pd.read_sql(f"SELECT * FROM selected_data", conn))
    df = pd.concat(dfs, ignore_index=True)
    
    # Add a new column summarizing the decoding parameters
    df.loc[:, "decoding_params"] = df[["seconds", "components", "sampling_hz"]].astype(str).agg("_".join, axis=1)

    # Filter out the parameter combination of interest
    df = df[df["decoding_params"] == combination]
    
    # Statsitically compare the decoding performances in 200 runs
    grouped = df.groupby(['subject', 'channel', 'decoding_params'])

    results = []
    for (subject, channel, decoding_params), group in grouped:
        # pivot or filter to get accuracy for shuffled=0 and shuffled=1 per run
        acc_0 = group[group['shuffled'] == 0].sort_values('run')['accuracy']
        acc_1 = group[group['shuffled'] == 1].sort_values('run')['accuracy']
        
        # Wilcoxon signed-rank test (paired samples)
        stat, p_value = wilcoxon(acc_0, acc_1)
        
        results.append({
            'subject': subject,
            'channel': channel,
            'decoding_params': decoding_params,
            'wilcoxon_stat': stat,
            'p_value': p_value,
            'mean_accuracy_shuffled_0': acc_0.mean(),
            'mean_accuracy_shuffled_1': acc_1.mean()
        })

    # create a result DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df[results_df["mean_accuracy_shuffled_0"] > 0.5]

    # Count number of unique channels per subject
    channel_counts = results_df.groupby('subject')['channel'].nunique().to_dict()

    # Apply Bonferroni correction per subject
    results_df['p_value_corrected'] = results_df.apply(
        lambda row:  1-min(row['p_value'] * channel_counts[row['subject']], 1.0),
        axis=1
    )

    # Prepare plotting
    fig, ax = plt.subplots(1, 1, figsize=(13, 8))

    # Plot the p-values
    sns.barplot(data=results_df, x="subject", y="p_value_corrected", hue="channel", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("p-value")
    ax.set_ylim([0.95, 1])

    # Adjust
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Save
    plt.savefig(f"figures/WM_decoding_stats_{combination}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    combination= "5_4_1"
    plot_WM_decoding(combination)
