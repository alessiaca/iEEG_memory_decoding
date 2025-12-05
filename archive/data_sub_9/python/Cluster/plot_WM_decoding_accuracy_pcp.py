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
import plotly.express as px


def plot_WM_decoding():

    # Load the decoding accuracies from folder 
    db_files = glob.glob(os.path.join('decoding_results/', '*.db'))
    dfs = []
    for db in db_files:
        with sqlite3.connect(db) as conn:
            dfs.append(pd.read_sql(f"SELECT * FROM selected_data", conn))
    df = pd.concat(dfs, ignore_index=True)
    
    # Keep only non-shuffled results and add a new column summarizing the decoding parameters
    df = df[df["shuffled"] == 0]
    
    # Calculate the mean accuracy across all runs 
    df = df.groupby(["subject", "channel", "seconds", "components", "sampling_hz"])["accuracy"].mean().reset_index()

    # Calculate the mean across channels/subjects
    df_mean = df.groupby(["seconds", "components", "sampling_hz"])["accuracy"].mean().reset_index()

    # 🔧 Ensure numeric types for Plotly
    for col in ["seconds", "components", "sampling_hz", "accuracy"]:
        df_mean[col] = pd.to_numeric(df_mean[col], errors="coerce")

    # Plot with color mapped to accuracy
    fig = px.parallel_coordinates(
        df_mean,
        dimensions=["seconds", "components", "sampling_hz"],
        color="accuracy",
        color_continuous_scale="Viridis",
        labels={col: col for col in df_mean.columns},
        range_color=[0.5, 0.6]
    )

    fig.update_layout(title="Parallel Coordinates Plot Colored by Accuracy")
    fig.show()


if __name__ == "__main__":
    plot_WM_decoding()