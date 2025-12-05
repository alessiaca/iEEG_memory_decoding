# Plot the WM decoding performance (for different parameter combinations)

import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns
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


def plot_WM_decoding():

    # Load the decoding accuracies from folder 
    db_files = glob.glob(os.path.join('decoding_results/', '*.db'))
    dfs = []
    for db in db_files:
        with sqlite3.connect(db) as conn:
            dfs.append(pd.read_sql(f"SELECT * FROM selected_data", conn))
    df = pd.concat(dfs, ignore_index=True)

    # Filter out subjects of interest
    subjects = [2, 3, 6, 7, 8, 9]
    df = df[df["subject"].isin(subjects)]
    
    # Add a new column summarizing the decoding parameters
    df.loc[:, "decoding_params"] = df[["seconds", "components", "sampling_hz"]].astype(str).agg("_".join, axis=1)

    # Compute Cohen's d for each subject x decoding_params
    df_d = (
        df.groupby(["subject", "channel", "decoding_params"])
        .apply(compute_cohens_d)
        .reset_index(name="cohens_d")
    )

    # Average across subjects for each decoding_params
    df_d_mean = df_d.groupby("decoding_params")["cohens_d"].mean().reset_index()
    df_d_mean = df_d_mean.sort_values(by="cohens_d", ascending=False)

    # Prepare plotting
    fig, axes = plt.subplots(4, 1, figsize=(13, 8))

    # Plot cohens' d
    ax = axes[0]
    sns.barplot(data=df_d_mean, x="decoding_params", y="cohens_d", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=7)
    ax.set_ylabel("Avg. Cohen's d")
    ax.set_xlabel("")

    # Plot the average accuracy across all channels
    ax = axes[1]
    df = df[df["shuffled"] == 0]
    df = df.groupby(["subject", "channel", "decoding_params"])["accuracy"].mean().reset_index()
    df_mean = df.groupby("decoding_params")["accuracy"].mean().reset_index()
    df_mean = df_mean.sort_values(by="accuracy", ascending=False)
    sns.barplot(data=df_mean, x="decoding_params", y="accuracy", ax=ax)
    ax.axhline(0.5, color="red")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("Average accuracy")

    # Plot the average accuracy of the maximum channel across all patients
    ax = axes[2]
    df_max = df.loc[df.groupby(["subject", "decoding_params"])["accuracy"].idxmax()]
    df_max = df_max.groupby("decoding_params")["accuracy"].mean().reset_index()
    df_max = df_max.sort_values(by="accuracy", ascending=False)
    sns.barplot(data=df_max, x="decoding_params", y="accuracy", ax=ax)
    ax.axhline(0.5, color="red")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=7)
    ax.set_ylabel("Average per-subject \nmaximum accuracy")

    # Plot the average ratio of channels > 0.5 across subject
    ax = axes[3]
    df_ratio = (
        df.groupby(["subject", "decoding_params"])
        .apply(lambda x: (x["accuracy"] > 0.5).mean())
        .reset_index(name="ratio_above_0_5")
    )
    df_ratio_mean = df_ratio.groupby("decoding_params")["ratio_above_0_5"].mean().reset_index()
    df_ratio_mean = df_ratio_mean.sort_values(by="ratio_above_0_5", ascending=False)
    sns.barplot(data=df_ratio_mean, x="decoding_params", y="ratio_above_0_5", ax=ax, order=df_ratio_mean["decoding_params"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("Avg. Ratio of Channels > 0.5")

    # Adjust
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Save
    plt.savefig("figures/WM_decoding.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_WM_decoding()