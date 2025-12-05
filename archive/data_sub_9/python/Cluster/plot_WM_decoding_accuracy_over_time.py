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


def compute_cohens_d(group):
    true = group[group["shuffled"] == 0]["accuracy"]
    perm = group[group["shuffled"] == 1]["accuracy"]

    n1, n2 = len(true), len(perm)
    if n1 > 1 and n2 > 1:
        s1 = np.std(true, ddof=1)
        s2 = np.std(perm, ddof=1)
        s_pooled = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
        if s_pooled > 0:
            d = (true.mean() - perm.mean()) / s_pooled
        else:
            d = np.nan
    else:
        d = np.nan

    return d


def compute_p_value(group):
    true = group[group["shuffled"] == 0]["accuracy"]
    perm = group[group["shuffled"] == 1]["accuracy"]
    p_value = (perm > float(true)).mean()
    return p_value


def plot_heatmaps(df_d_mean, target, name):
    unique_components = sorted(df_d_mean["components"].unique())
    n_cols = 2
    n_rows = int(np.ceil(len(unique_components) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8), constrained_layout=True)
    axes = axes.flatten()
    vmin = df_d_mean[target].min()
    vmax = df_d_mean[target].max()

    for i, comp in enumerate(unique_components):
        ax = axes[i]
        data_subset = df_d_mean[df_d_mean["components"] == comp]
        pivot_table = data_subset.pivot(index="seconds", columns="sampling_hz", values=target)

        sns.heatmap(pivot_table, ax=ax, annot=True, fmt=".2f", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Components = {comp}")
        ax.set_ylabel("Seconds")
        ax.set_xlabel("Sampling Hz")

    # Hide any extra subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(name, fontsize=14)


def plot_WM_decoding(normalize):

    palettes = [["#434346", "#706E6E"], ['#0000FF', "#706E6E"], ['#C34290', "#706E6E"], ['#0ABAB5', "#706E6E"]]

    # Load the decoding accuracies
    folder_path = "decoding_results_2 (8)"

    # Get list of all CSV files
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # Load and concatenate all CSVs
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Filter out subjects of interest
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    df = df[df["subject"].isin(subjects)]

    # Compute p value
    #df_p = (df.groupby(["subject", "channel", "period"]).apply(compute_p_value).reset_index(name="p-value"))

    df_mean_run = df# df.groupby(['subject', 'period', "channel", "shuffled", "perm"])["accuracy"].mean().reset_index()
    #df_mean_run = df_mean_run[df_mean_run["accuracy"] > 0.5]
    #df_mean_run = df.groupby(['subject', 'period'])["accuracy"].max().reset_index()
    fig, axes = plt.subplots(3, 3, figsize=(10,10))
    axes = axes.flatten()
    for subject, ax in zip(subjects, axes):
        sns.lineplot(
            data=df_mean_run[(df_mean_run.subject == subject)],
            x="period",
            y="accuracy",
            hue="shuffled",
            ax=ax,
            errorbar="sd"                  # Shaded area = ±1 std
    )
        ax.axhline(0.5)
        ax.axvline(10)
        ax.axvline(30)
        ax.axvline(60)
    plt.show()
   

if __name__ == "__main__":
    plot_WM_decoding(True)
    plot_WM_decoding(False)