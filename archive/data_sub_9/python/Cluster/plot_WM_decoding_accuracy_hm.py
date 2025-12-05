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

    res = wilcoxon(true, perm)
    return res.pvalue


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


def plot_WM_decoding():

    # Load the decoding accuracies from folder 
    db_files = glob.glob(os.path.join('decoding_results/', '*.db'))
    dfs = []
    for db in db_files:
        with sqlite3.connect(db) as conn:
            dfs.append(pd.read_sql(f"SELECT * FROM selected_data", conn))
    df = pd.concat(dfs, ignore_index=True)

    # Filter out subjects of interest
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    df = df[df["subject"].isin(subjects)]

    # Compute Cohen's d for each subject x decoding_params
    df_d = (
        df.groupby(["subject", "channel", "seconds", "components", "sampling_hz"])
        .apply(compute_cohens_d)
        .reset_index(name="cohens_d")
    )

    # Average across subjects for each decoding_params
    df_d_mean = df_d.groupby(["seconds", "components", "sampling_hz"])["cohens_d"].mean().reset_index()
    df_d_mean = df_d_mean.sort_values(by="cohens_d", ascending=False)

    # Plot cohens' d as heatmap
    plot_heatmaps(df_d_mean, target="cohens_d", name="Cohen's d (shuffled-not shuffled)")

    # Save
    plt.savefig("figures/WM_decoding_d_hm.png", dpi=300)

    # Keep relevant accuracy values
    df_acc = df[df["shuffled"] == 0].groupby(
        ["subject", "channel", "seconds", "components", "sampling_hz"]
    )["accuracy"].mean().reset_index()

    # Compute p-values
    df_p = (
        df.groupby(["subject", "channel", "seconds", "components", "sampling_hz"])
        .apply(compute_p_value)
        .reset_index(name="p_value")
    )

    # Merge accuracy into df_p
    df_p = pd.merge(df_p, df_acc, on=["subject", "channel", "seconds", "components", "sampling_hz"])

    df_p_pct = (
        df_p.assign(sig=lambda d: (d["p_value"] < 0.05) & (d["accuracy"] > 0.5))
            .groupby(["seconds", "components", "sampling_hz"])["sig"]
            .mean()
            .reset_index(name="sig_pct")
    )
    df_p_pct["sig_pct"] *= 100

    # Plot cohens' d as heatmap
    plot_heatmaps(df_p_pct, target="sig_pct", name="Percentage signficant")

    # Save
    plt.savefig("figures/WM_decoding_p_hm.png", dpi=300)

    # Plot the average decoding accuracy as heatmap
    df = df[df["shuffled"] == 0]
    df = df.groupby(["subject", "channel", "seconds", "components", "sampling_hz"])["accuracy"].mean().reset_index()
    df_mean = df.groupby(["seconds", "components", "sampling_hz"])["accuracy"].mean().reset_index()
    plot_heatmaps(df_mean, target="accuracy", name="Decoding Accuracy")

    # Save
    plt.savefig("figures/WM_decoding_mean_acc_hm.png", dpi=300)

    # Plot the maximum decoding accuracy across subjects as heatmap
    df_max = df.loc[df.groupby(["subject", "seconds", "components", "sampling_hz"])["accuracy"].idxmax()]
    df_max = df_max.groupby(["seconds", "components", "sampling_hz"])["accuracy"].mean().reset_index()
    plot_heatmaps(df_max, target="accuracy", name="Max Decoding Accuracy")

    # Save
    plt.savefig("figures/WM_decoding_max_acc_hm.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    plot_WM_decoding()