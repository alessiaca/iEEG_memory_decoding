# Plot the correlation of the empirical accuracy and the similarity to the optimal functional connectivity map

import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import matplotlib
import os
import statsmodels.formula.api as smf
import seaborn as sns
import glob
matplotlib.use("TkAgg")


def plot_func_conn_individual(combination):

    # Load the decoding accuracies from folder 
    db_files = glob.glob(os.path.join('functional_connectivity_results (6)/', '*.db'))
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

    # Get the combination of interest
    df = df[df["func_conn_params"] == combination]
    df = df.dropna(subset=["similarity", "accuracy"])

    # Calculate the spearman correlation between the similarity and the decoding accuracy 
    r, p = spearmanr(df['similarity'], df['accuracy'])

    # Scatter + regression line
    fig, axes = plt.subplots(1, 1, figsize=(1.5, 1))
    ax = axes
    fontsize = 6
    sns.regplot(data=df, x='similarity', y='accuracy', ax=ax, scatter_kws={'s':5}, color="grey")
    ax.set_title(f"r={np.round(r, 3)}, p={np.round(p, 3)}", fontsize=fontsize+1)
    ax.set_xlabel("Similarity to R-map", fontsize=fontsize+1)
    ax.set_ylabel("Decoding accuracy", fontsize=fontsize+1)
    ax.tick_params(axis='x', labelsize=fontsize)   # x-axis ticks
    ax.tick_params(axis='y', labelsize=fontsize)   # y-axis ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(bottom=0.2, left=0.2)

    # Adjust
    plt.subplots_adjust(left=0.2, bottom=0.2)

    # Save
    plt.savefig(f"figures/func_conn_{combination}.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    masks = ["None", "mem", "rFo", "rnu"]
    for mask in masks:
        combination = f"5_3_1_5.0_false_nan_{mask}"
        plot_func_conn_individual(combination)