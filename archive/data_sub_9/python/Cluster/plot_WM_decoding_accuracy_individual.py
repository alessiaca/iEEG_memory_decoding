# Plot the WM decoding performance (for s specific parameter combination)

import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns
import glob
matplotlib.use("TkAgg")


def plot_WM_decoding_individual(combination):

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

    # Prepare plotting
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    # Plot the average accuracy across all channels
    ax = axes[0]
    df_mean = df.groupby(["subject", "channel"])["accuracy"].max().reset_index()
    sns.barplot(data=df_mean, x="subject", y="accuracy", hue="channel", ax=ax)
    ax.axhline(0.5, color="red")
    ax.set_xlabel("")
    ax.set_ylabel("Maximum accuracy")
    ax.set_ylim([0.2, 0.8])

    # Plot the average accuracy of the maximum channel across all patients
    ax = axes[1]
    df_ind = df[df["decoding_params"] == combination]
    sns.barplot(data=df_ind, x="subject", y="accuracy", hue="channel", ax=ax)
    ax.axhline(0.5, color="red")
    ax.set_ylabel(f"{combination} accuracy")
    ax.set_ylim([0.2, 0.8])

    # Adjust
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Save
    plt.savefig(f"figures/WM_decoding_{combination}.png", dpi=300)
    plt.show()


if __name__ == "__main__":

    combination = "2_3_2"
    plot_WM_decoding_individual(combination)