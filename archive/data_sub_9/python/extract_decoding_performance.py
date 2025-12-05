import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns
matplotlib.use("TkAgg")

# Define parameters
referencing = "none"
Hz = 5
length = 1000
n_runs = 201
model_name = f"{referencing}_Hz_{Hz}_length_{length}_runs_{n_runs}"

# Prepare plotting
fig, ax = plt.subplots(figsize=(13, 6))
width = 0.3

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

    df = df[df["run"] > 0]

    # Compute mean accuracy
    mean_decoding = df.groupby(["subject", "channel"], as_index=False)["accuracy"].mean()

    # Precompute all channel-level decoding arrays
    channel_decodings = {
        channel: group["accuracy"].values
        for channel, group in df.groupby("channel")
    }

    all_subjects_data.append(mean_decoding)

# Combine and save results
df = pd.concat(all_subjects_data, ignore_index=True)
df.to_csv(f"../../Decoding_results/{model_name}.csv", index=False)

# Plot the results
sns.barplot(data=df, x="subject", y="accuracy", hue="channel")
plt.axhline(0.5)

# Compute annotation text: count of channels > 0.5 / total channels per subject
summary = df.groupby("subject").apply(
    lambda g: f"{(g.accuracy > 0.5).sum()}/{len(g)}"
).reset_index(name="count_text")

# Place the annotations above the grouped bars
# Get the position of the bars (x-coordinates) for each subject
xticks = ax.get_xticks()
for i, (_, row) in enumerate(summary.iterrows()):
    ax.text(
        xticks[i],                 # x position (subject group center)
        1.02,                      # y position above the plot (adjust as needed)
        row["count_text"], 
        ha='center', va='bottom', 
        fontsize=10, fontweight='bold'
    )

# Customize plot
plt.title("Accuracy by Channel and Subject")
plt.ylabel("Accuracy")
plt.ylim(0, 1.1)  # leave space for the text
plt.legend(title="Channel", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(f"../../Figures/{model_name}_performance.png", dpi=300)
plt.show()

