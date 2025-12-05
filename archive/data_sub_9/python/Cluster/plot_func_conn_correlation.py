# Plot the correlation of the empirical accuracy and the similarity to the optimal functional connectivity map

import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import matplotlib
import os
import seaborn as sns
import glob
matplotlib.use("TkAgg")


def plot_func_conn():

    # Load the decoding accuracies from folder 
    db_files = glob.glob(os.path.join('functional_connectivity_results/', '*.db'))
    dfs = []
    for db in db_files:
        with sqlite3.connect(db) as conn:
            dfs.append(pd.read_sql(f"SELECT * FROM selected_data", conn))
    df = pd.concat(dfs, ignore_index=True)
    
    # Round and shorten for readability
    df["subject_mode"] = df["subject_mode"].round(2)
    df["mask"] = df["mask"].str[:3]  # Shorten mask names for readability
    # Add a new column summarizing the functional connectivity analysis parameters
    df.loc[:, "func_conn_params"] = df[["decoding_mode", "subject_mode", "flip_mode", "overlap", "mask"]].astype(str).agg("_".join, axis=1)

    # Calculate the spearman correlation between the similarity and the decoding accuracy for each parameter combination
    df_results = df.groupby("func_conn_params").apply(lambda x: pd.Series(spearmanr(x["accuracy"], x["similarity"]), index=["spearman_r", "p_value"])).reset_index()# if func_conn_params is index
    df_results["significant"] = df_results["p_value"] < 0.05
    palette = df_results["significant"].map({True: "red", False: "black"}).tolist()

    # Prepare plotting
    fig, ax = plt.subplots(1, 1, figsize=(13, 8))

    # Plot the average accuracy across all channels
    #ax = axes[0]
    sns.barplot(data=df_results, x="func_conn_params", y="spearman_r", palette=palette, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("Average spearman r")

    # Adjust
    plt.subplots_adjust(bottom=0.4)

    # Save
    plt.savefig("figures/func_conn.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_func_conn()