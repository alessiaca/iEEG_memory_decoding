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
n_runs = 200
model_name = f"{referencing}_n_runs_{n_runs}"

# Load database
db_path = f"../../Decoding_results/{model_name}.db"
with sqlite3.connect(db_path) as conn:
    df = pd.read_sql("SELECT * FROM selected_data", conn)

# Drop nan values
df = df.dropna(subset=['accuracy'])

# Add channel/subject column
df["combination"] = df["seconds"].astype(str) + "-" + df["components"].astype(str) + "-" + df["sampling_hz"].astype(str)

# Compute the mean decoding accuracy 
df_mean_acc = df.groupby(['combination','subject','channel']).agg({
    'accuracy': 'mean'
    # 'other_col': 'first'  # and so on...
    }).reset_index()

# Step 2: Filter only entries where mean accuracy > 0.5
df_above_thresh = df_mean_acc[df_mean_acc['accuracy'] > 0.5]

# Step 3: Count how many channels per subject exceed threshold per combination
counts_above = df_above_thresh.groupby(['combination', 'subject']).size().reset_index(name='n_channels_above')

# Step 4: Count total channels per subject per combination (to get denominator)
total_channels = df_mean_acc.groupby(['combination', 'subject']).size().reset_index(name='n_channels_total')

# Step 5: Merge and compute ratio
ratios = pd.merge(counts_above, total_channels, on=['combination', 'subject'])
ratios['ratio'] = ratios['n_channels_above'] / ratios['n_channels_total']

# Step 6: Average the ratio across subjects for each combination
mean_ratios = ratios.groupby('combination')['ratio'].mean().reset_index()

# Step 7: Get combination(s) with highest average ratio
best_combination = mean_ratios.sort_values('ratio', ascending=False).head(1)

print(best_combination)

# Get the maximum accuracy per subject/channel
idx = df_mean_acc.groupby(['subject', 'channel'])['accuracy'].idxmax()
df_max_acc = df_mean_acc.loc[idx].reset_index(drop=True)

max_comb = df_max_acc.groupby("combination").nunique()["channel"].idxmax()
max_comb =  best_combination.combination.iloc[0]

# Plot the maximum accuracy from the grid search
fig, axes = plt.subplots(2, 1, figsize=(13, 6))
width = 0.3
sns.barplot(data=df_max_acc, x="subject", y="accuracy", hue="channel", ax=axes[0])
axes[0].axhline(0.5)

# Plot the accuracy using the combination which gives the best decoding performances for the majority
best_comb = df_mean_acc[df_mean_acc["combination"] == max_comb]
sns.barplot(data=best_comb, x="subject", y="accuracy", hue="channel", ax=axes[1])
axes[1].axhline(0.5)
axes[1].set_ylabel(f"Accuracy {max_comb}")

# Compute annotation text: count of channels > 0.5 / total channels per subject
summary = df_max_acc.groupby("subject").apply(
    lambda g: f"{(g.accuracy > 0.5).sum()}/{len(g)}"
).reset_index(name="count_text")

# Place the annotations above the grouped bars
# Get the position of the bars (x-coordinates) for each subject
xticks = axes[0].get_xticks()
for i, (_, row) in enumerate(summary.iterrows()):
    axes[0].text(
        xticks[i],                 # x position (subject group center)
        0.7,                      # y position above the plot (adjust as needed)
        row["count_text"], 
        ha='center', va='bottom', 
        fontsize=10, fontweight='bold'
    )

# Compute annotation text: count of channels > 0.5 / total channels per subject
summary = best_comb.groupby("subject").apply(
    lambda g: f"{(g.accuracy > 0.5).sum()}/{len(g)}"
).reset_index(name="count_text")

# Place the annotations above the grouped bars
# Get the position of the bars (x-coordinates) for each subject
xticks = axes[1].get_xticks()
for i, (_, row) in enumerate(summary.iterrows()):
    axes[1].text(
        xticks[i],                 # x position (subject group center)
        0.7,                      # y position above the plot (adjust as needed)
        row["count_text"], 
        ha='center', va='bottom', 
        fontsize=10, fontweight='bold'
    )

# Save the accuracies

plt.figure()
best_comb = df_max_acc
best_comb = best_comb[best_comb["accuracy"] > 0.5]
sns.scatterplot(y=best_comb.index, x=best_comb["accuracy"])

df_max_acc["channel"] = df_max_acc["channel"] + 1
df_max_acc.to_csv(f"../../Decoding_results/{model_name}.csv", index=False)

plt.show()
