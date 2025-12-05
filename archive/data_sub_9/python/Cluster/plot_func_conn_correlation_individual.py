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
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    sns.regplot(data=df, x='similarity', y='accuracy', ax=ax, scatter_kws={'s':50})
    ax.set_title(f"r={np.round(r, 3)}, p={np.round(p, 3)}, {combination}")

    # Plot the subject-wise correlation (LME Model)
    ax = axes[1]
    df_lmm = df.dropna(subset=["similarity"])
    model = smf.mixedlm(f"accuracy ~ similarity", df_lmm, groups=df_lmm["subject"])
    result = model.fit()
    print(result.summary())
    subjects = df_lmm["subject"].unique()
    colors = sns.color_palette("husl", len(subjects))
    for i, subj in enumerate(subjects):
        df_sub = df_lmm[df_lmm["subject"] == subj]
        sns.regplot(
            x="accuracy", y="similarity", data=df_sub,
            scatter_kws={'s': 30, 'alpha': 0.6},
            line_kws={"linewidth": 1.5},
            color=colors[i],
            label=f"Subject {subj}", 
            ax=ax
        )
    ax.set_xlabel("Decoding Performance", fontsize=12)
    ax.set_ylabel(f"Connectivity Similarity", fontsize=12)
    ax.set_title(f"LME p = {np.round(result.pvalues.similarity, 3)}")
    plt.legend(title="Subjects", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    # Adjust
    plt.subplots_adjust(right=0.8, wspace=0.4, hspace=0.4)

    # Save
    plt.savefig(f"figures/func_conn_{combination}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    masks = ["None", "mem", "rFo", "rnu"]
    for mask in masks:
        combination = f"5_3_1_5.0_false_nan_{mask}"
        plot_func_conn_individual(combination)