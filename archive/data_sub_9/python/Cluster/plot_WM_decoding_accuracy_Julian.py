# Plot the WM decoding performance (for different parameter combinations)

import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns
from scipy.stats import wilcoxon
import glob
matplotlib.use("TkAgg")


def plot_WM_decoding():

    # Load the decoding accuracies from folder 
    db_files = glob.glob(os.path.join('decoding_results_2 (19)/', '*.db'))
    dfs = []
    for db in db_files:
        with sqlite3.connect(db) as conn:
            dfs.append(pd.read_sql(f"SELECT * FROM selected_data", conn))
    df = pd.concat(dfs, ignore_index=True)

    # Filter out subjects of interest
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    df = df[df["subject"].isin(subjects)]

    # Keep only accuracies of interest
    df = df[df.sampling_hz == 3]

    # Keep relevant accuracy values
    df_acc = df[df["shuffled"] == 0].groupby(
        ["subject", "channel"]
    )["accuracy"].mean().reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    fontsize = 7
    sns.violinplot(data=df_acc, x="subject", y="accuracy", ax=ax, color="#C34290", inner=None, linewidth=0)
    sns.swarmplot(x="subject", y="accuracy", data=df_acc, 
              color="black", size=1.5, alpha=0.7)
    ax.set_ylabel("5-fold cross-validation \ntest accuracy", fontsize=fontsize+1)
    plt.axhline(0.5, color="grey", linestyle="--")
    ax.set_xlabel("Subject", fontsize=fontsize+1)
    ax.tick_params(axis='x', labelsize=fontsize)   # x-axis ticks
    ax.tick_params(axis='y', labelsize=fontsize)   # y-axis ticks
    plt.subplots_adjust(bottom=0.2, left=0.2)

    # Save
    plt.savefig("figures/decoding_accuracies.pdf", dpi=300)

    plt.show()


if __name__ == "__main__":
    plot_WM_decoding()