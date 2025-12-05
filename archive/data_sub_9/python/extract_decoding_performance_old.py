import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.stats import wilcoxon
matplotlib.use("TkAgg")

# Define parameters
referencing = "none"
Hz = 5
length = 1000
n_runs = 200
model_name = f"{referencing}_Hz_{Hz}_length_{length}_runs_{n_runs}"

# Subjects and output container
subjects = np.arange(1, 10)
all_subjects_data = []

for subject in subjects:
    db_path = f"../../Decoding_results/{model_name}/subject_{subject}_{referencing}_Hz_{Hz}_length_{length}.db"
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        continue

    # Load data
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM selected_data", conn)

    # Remove channels with zero variance
    channel_variances = df.groupby("channel")["accuracy"].var()
    valid_channels = channel_variances[channel_variances > 0].index
    df = df[df["channel"].isin(valid_channels)]

    df_perm = df[df["run"] < 0]
    df = df[df["run"] > 0]

    # Compute mean accuracy
    mean_decoding = df.groupby(["subject", "channel"], as_index=False)["accuracy"].mean()
    mean_decoding_perm = df_perm.groupby(["subject", "channel"], as_index=False)["accuracy"].mean()

    # Precompute all channel-level decoding arrays
    channel_decodings = {
        channel: group["accuracy"].values
        for channel, group in df.groupby("channel")
    }
    channel_decodings_perm = {
        channel: group["accuracy"].values
        for channel, group in df_perm.groupby("channel")
    }

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(mean_decoding))
    width = 0.3

    # Plot bars individually to allow per-bar alpha
    # Store p-values
    p_values = []

    for i, (ch, acc, acc_perm) in enumerate(zip(mean_decoding["channel"], mean_decoding["accuracy"], mean_decoding_perm["accuracy"])):
        # Compute significance
        res = wilcoxon(channel_decodings[ch], channel_decodings_perm[ch])
        p_values.append(res.pvalue)

        """if acc > 0.59 or 1 - acc > 0.59:
            color = "magenta"
        elif res.pvalue < 0.05 / len(mean_decoding) and acc > acc_perm:
            color = "red"
        elif res.pvalue < 0.05 / len(mean_decoding) and acc < acc_perm:
            color = "green"
        else:
            color = "black"""
        if acc > 0.5:
            color = "red"
        else:
            color = "black"

        ax.bar(i, acc, width=width, color=color, alpha=1.0)
        ax.bar(i + width, acc_perm, width=width, color="grey", alpha=1.0)

        ax.axhline(0.59)

        ax.scatter([i] * len(channel_decodings[ch]), channel_decodings[ch], s=0.3, color="black")
        ax.scatter([i + width] * len(channel_decodings_perm[ch]), channel_decodings_perm[ch], s=0.3, color="black")

        """if 1-acc > 0.59:
            mean_decoding.loc[i, "accuracy"] = 1 - acc"""


    #ax.axhline(0.5)
    ax.set_xlabel('Channels')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Decoding Performance for subject {subject} model {model_name}', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(mean_decoding["channel"], fontsize=9, rotation=45, ha="right")
    ax.tick_params(axis='x', labelsize=9)
    ax.legend(["Chance Level", "Mean Accuracy", "Individual Trials"])
    ax.set_ylim([-0.1, 0.7])

    plt.savefig(f"../../Figures/{model_name}_performance_{subject}_nruns_{n_runs}.png", dpi=300)

    # Store result
    mean_decoding["p_value"] = p_values
    all_subjects_data.append(mean_decoding)

# Combine and save results
combined_data = pd.concat(all_subjects_data, ignore_index=True)
combined_data.to_csv(f"../../Decoding_results/{model_name}.csv", index=False)

plt.show()
