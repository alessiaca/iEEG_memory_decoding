# Plot the WM decoding performance (for different parameter combinations)

import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.stats import spearmanr
import seaborn as sns
import glob
matplotlib.use("TkAgg")

def plot_heatmaps(df, target, name, group_params, pivot_index, pivot_columns, alpha=0.05):
    """
    Plots a grid of heatmaps based on groupings in the DataFrame.

    Parameters:
        df (pd.DataFrame): The data.
        target (str): The column to visualize.
        name (str): Title of the entire figure.
        group_params (list[str]): Two column names for subplot rows and columns.
        pivot_index (str): Column name to use for heatmap rows.
        pivot_columns (str): Column name to use for heatmap columns.
        alpha (float): Significance threshold for marking values.
    """
    if len(group_params) != 2:
        raise ValueError("group_params must contain exactly two elements (for rows and columns of subplots)")

    row_param, col_param = group_params
    row_values = sorted(df[row_param].dropna().unique())
    col_values = sorted(df[col_param].dropna().unique())

    n_rows = len(row_values)
    n_cols = len(col_values)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), constrained_layout=True)
    axes = np.array(axes).reshape(n_rows, n_cols)

    vmin = df[target].min()
    vmax = df[target].max()

    for i, row_val in enumerate(row_values):
        for j, col_val in enumerate(col_values):
            ax = axes[i, j]
            subset = df[(df[row_param] == row_val) & (df[col_param] == col_val)]

            values = subset.pivot(index=pivot_index, columns=pivot_columns, values=target)
            pvals = subset.pivot(index=pivot_index, columns=pivot_columns, values="p_value")

            annotations = values.copy().astype(str)
            for row in annotations.index:
                for col in annotations.columns:
                    val = values.loc[row, col]
                    pval = pvals.loc[row, col]
                    if pd.notna(pval) and pval < alpha:
                        annotations.loc[row, col] = f"{val:.2f}*"
                    elif pd.notna(val):
                        annotations.loc[row, col] = f"{val:.2f}"
                    else:
                        annotations.loc[row, col] = ""

            sns.heatmap(values, ax=ax, annot=annotations, fmt="", cmap="viridis", vmin=vmin, vmax=vmax, cbar=True)
            ax.set_title(f"{row_param}: {row_val}, {col_param}: {col_val}")
            ax.set_ylabel(pivot_index)
            ax.set_xlabel(pivot_columns)

    plt.suptitle(name, fontsize=14)
    plt.show()



def plot_func_conn_hm():

    # Load the decoding accuracies from folder 
    folder_path = 'functional_connectivity_results/'
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    # Load and concatenate all CSVs
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    
    # Round and shorten for readability
    df["subject_mode"] = df["subject_mode"].round(2)
    df["mask"] =  df["mask"].str[:3].astype(str)
    df["overlap_thresh"] = df["overlap_thresh"].astype(str)
    df["accuracy_thres"] = df["accuracy_thres"].astype(str)

    df = df.dropna(subset=["similarity", "accuracy"])

    # Calculate the spearman correlation between the similarity and the decoding accuracy for each parameter combination
    df_results = df.groupby(["sampling_hz", "accuracy_thres", "subject_mode", "flip_mode", "overlap_thresh", "mask"]).apply(lambda x: pd.Series(spearmanr(x["accuracy"], x["similarity"]), index=["spearman_r", "p_value"])).reset_index()# if func_conn_params is index

    # Plot spearman r as heatmap
    # excluding subject 1,4
    df_tmp = df_results[df_results["subject_mode"] == 5.0]
    plot_heatmaps(
    df=df_tmp,
    target="spearman_r",
    name="Correlation Heatmaps",
    group_params=["sampling_hz", "mask"],
    pivot_index="overlap_thresh",
    pivot_columns="flip_mode"
    )
    # Save
    plt.savefig("figures/func_conn_561_hm.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    plot_func_conn_hm()