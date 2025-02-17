
# Plot the performance of the decoding
# One overview plot with all subjects displaying the p values
# One plot per subject with the performances

import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

subjects = np.arange(1, 10)
fig, axes = plt.subplots(1, 1, figsize=(5, 5))

for subject in subjects:

    # Load db with subject performances to dataframe
    conn = sqlite3.connect(f"results/memory_results (4)/subject_{subject}.db")
    df = pd.read_sql("SELECT * FROM selected_data", conn)

    fig_sub, axes_sub = plt.subplots(1, 1, figsize=(15, 5))

    # Loop over each channel
    sorted_channels = sorted(df["ch"].unique(), key=lambda x: int(x.split("_")[0][2:]))
    ps = []
    perfs = []
    for i, ch in enumerate(sorted_channels):

        df_ch = df[df["ch"] == ch]

        # Calculate the p-value
        true_perf = np.array(df_ch[df_ch["permutation"] == -1]["performance_test"])
        permuted_perf = np.array(df_ch[df_ch["permutation"] != -1]["performance_test"])
        p = np.sum(permuted_perf > true_perf) / len(permuted_perf)
        ps.append(p)
        perfs.append(true_perf[0])

        # Plot the ture performance as bar plot and the permuted performance as scatter
        color = "grey" if p > 0.05 else "red"
        axes_sub.bar(ch, true_perf, color=color)
        axes_sub.scatter([ch]*len(permuted_perf), permuted_perf, color="black", s=1)

    # Adjust subject plot
    axes_sub.set_ylabel("accuracy")
    axes_sub.set_xlabel("Channel")
    axes_sub.set_xticklabels(axes_sub.get_xticklabels(), rotation=90, fontsize=7)
    axes_sub.set_title(f"Subject {subject}: 4-8 workload test classification, 10-fold- crossvalidation, 50 permutations")

    # Save the plot
    fig_sub.savefig(f"plot_perf_{subject}.png")
    #plt.close(fig_sub)

    # Plot as scatter (at the subject location with small jitter)
    axes.scatter([subject]*len(ps), ps, color="black", s=1)

    # Save csv with channel name, performance and p value
    df = pd.DataFrame({"ch": sorted_channels, "performance": perfs, "p": ps})
    df.to_csv(f"perf_{subject}.csv", index=False)

# Adjust plot
axes.axhline(0.05, color="red")
axes.text(1, 0.07, "p = 0.05", color="red")
axes.set_ylabel("p-value (Permutation)")
axes.set_xlabel("Subject")
axes.set_title("4-8 workload test classification, 10-fold- crossvalidation, 50 permutations")

# Save the plot
fig.savefig("plot_perf.png")

plt.show()

