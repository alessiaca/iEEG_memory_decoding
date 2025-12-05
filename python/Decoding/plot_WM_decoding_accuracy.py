# Plot the WM decoding performance (for different parameter combinations)

from matplotlib import cm
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

    perm = np.array(perm)          # shape (n_perm,)
    true = float(true)            # observed statistic

    n_perm = len(perm)
    count = np.sum(perm >= true)  # one-sided: count permutations >= observed
    p_value = (count + 1) / (n_perm + 1)
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


def plot_WM_decoding():

    # Load the decoding accuracies
    folder_path = "Results"

    # Get list of all CSV files
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # Load and concatenate all CSVs
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Filter out subjects of interest
    subjects = np.arange(1, 16)
    df = df[df["subject"].isin(subjects)]

    # Plot the accuracy values as violin plots
    df_p = (df.groupby(["subject", "channel", "sampling_hz"]).apply(compute_p_value).reset_index(name="p-value"))

    # 1. Merge p-values into mean accuracy dataframe
    df = df[df["shuffled"] == 0]
    df_merged = pd.merge(df, df_p, on=['subject', 'channel', 'sampling_hz'], how='left')

    # save as csv
    df_merged.to_csv(f"WM_load_decoding_performance.csv", index=False)

    # Keep relevant accuracy values
    df_acc = df_merged[df_merged["shuffled"] == 0].groupby(
        ["subject", "channel"]
    )["p-value"].mean().reset_index()

    _, ax = plt.subplots(1, 1, figsize=(4, 3))
    fontsize = 7
    sns.violinplot(data=df_acc, x="subject", y="p-value", ax=ax, color="#C34290", inner=None, linewidth=0)
    sns.swarmplot(x="subject", y="p-value", data=df_acc, 
              color="black", size=1.5, alpha=0.7)
    ax.set_ylabel("5-fold cross-validation \ntest accuracy", fontsize=fontsize+1)
    plt.axhline(0.5, color="grey", linestyle="--")
    ax.set_xlabel("Subject", fontsize=fontsize+1)
    ax.tick_params(axis='x', labelsize=fontsize)   # x-axis ticks
    ax.tick_params(axis='y', labelsize=fontsize)   # y-axis ticks
    plt.subplots_adjust(bottom=0.2, left=0.2)

    # Save dataframe
    df_acc.to_csv("decoding_accuracies.csv", index=False)

    """plt.figure()
    acc = df_acc.accuracy.to_numpy()
    plt.hist(acc, bins=100, color="#53333F")
    plt.savefig("../../../figures/decoding_accuracies_hist.pdf", dpi=300, transparent=True)
    plt.show()

    plt.figure()
    cmap = cm.Blues
    sizes = df_acc.groupby("subject")["channel"].nunique().to_numpy()
    colors = cmap(np.linspace(0.3, 1, len(sizes)))
    plt.title(f"{sum(sizes)} channels across {len(sizes)} subjects")
    plt.pie(sizes, colors=colors)
    plt.savefig("../../../figures/decoding_accuracies_piechart.pdf", dpi=300, transparent=True)
    plt.show()"""

    _, axes = plt.subplots(1, 3, figsize=(12, 6))
    for i_period, period in enumerate(df.sampling_hz.unique()):
        df_tmp = df_merged[df_merged.sampling_hz == period]
        ax = axes[i_period]
        
        # 2. Plot violin plot (hue=shuffled, split)
        vp = sns.violinplot(data=df_tmp, x="subject", y="p-value", ax=ax, cut=0, inner=None)
        #axes[i_period].set_ylim([0.3, 0.8])
        axes[i_period].set_title(f"Sampling Hz: {period}")
        for violin in vp.collections:
            violin.set_edgecolor("#53333F")   # set edge color
            violin.set_linewidth(1.5)       # set edge width
            violin.set_facecolor("#53333F")  # optional: change fill color
        # Disable axis right and upper
        sns.despine(ax=ax)
        
        # 3. Overlay points manually to control color
        # For each (subject, shuffled), plot points colored by p-value
        
        unique_subjects = df_tmp['subject'].unique()
        shuffled_values = df_tmp['shuffled'].unique()
        
        # Map hue categories to offsets (for split violins)
        hue_order = sorted(shuffled_values)
        offset_dict = {hue_order[0]: -0.15, hue_order[1]: 0.15} if len(hue_order) == 2 else {h: 0 for h in hue_order}
        
        for i, subject in enumerate(sorted(unique_subjects)):
            for shuffled in hue_order:
                subset = df_tmp[(df_tmp['subject'] == subject) & (df_tmp['shuffled'] == shuffled)]
                if subset.empty:
                    continue
                
                # x position base + offset for split violin
                x_base = i
                x_offset = offset_dict[shuffled]
                
                # Jitter points horizontally within +/- 0.1
                jitter = np.random.uniform(-0.1, 0.1, size=len(subset))
                x_positions = x_base + x_offset + jitter
                
                # Set color based on p-value < 0.05: red if significant else black
                colors = ['#e72512' if p < 0.05 else 'white' for p,acc in zip(subset['p-value'],subset['accuracy'])]

                ax.scatter(x_positions, subset['p-value'], color=colors, alpha=0.7, s=10, edgecolor='w', linewidth=0)
        
        ax.axhline(0.5, color="black")
        ax.set_xlabel("Subject", fontsize=fontsize+1)
        ax.set_ylabel("Average test accuracy", fontsize=fontsize+1)
        ax.tick_params(axis='x', labelsize=fontsize-1)   # x-axis ticks
        ax.tick_params(axis='y', labelsize=fontsize)   # y-axis ticks
    plt.savefig("../../../figures/decoding_accuracies_all.pdf", dpi=300, transparent=True)
    plt.show()


if __name__ == "__main__":
    plot_WM_decoding()