# Extract features for each subject

import py_neuromodulation as nm
import numpy as np
import pandas as pd
from scipy.stats import zscore, ttest_ind
from utils import compute_p_tfr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

referencing = "none"
length = 1000
Hz = 5

subjects = np.arange(1, 10)

for subject in subjects:

    # Load the features
    features = pd.read_csv(f"../../Features/subject_{subject}_{referencing}_Hz_{Hz}_length_{length}/subject_{subject}_{referencing}_Hz_{Hz}_length_{length}_FEATURES.csv")

    # Load the label (set size)
    df = pd.read_csv(f'../../Processed_data/merged/Data_Subject_0{subject}.csv')
    set_size = df["Set size"].to_numpy()

    # Loop over channels
    for ch in range(64):
        ch_name = f'CH{ch}_'
        columns = [column for column in features.columns if ch_name in column]

        if len(columns) > 0:

            features_long = []
            for i in range(0, int(np.max(features.time) / 6000)):
                features_epoch = features[(features.time < 6000*(i+1)) & (features.time > 6000*i+1000)]
                # Select only the columns of interest
                features_epoch = features_epoch[columns].to_numpy()
                # Normalize by calculating percent in respect to first entry, keep only the last 2 seconds
                features_epoch_norm = (features_epoch[int(-1.5 * Hz):, :] - features_epoch[0, :]) / features_epoch[0, :]
                features_long.append(features_epoch_norm.mean(axis=0))

            features_long = np.array(features_long)

            """fig, ax = plt.subplots(1, 1)
            cax = ax.imshow(zscore(features_long.T[:, :100], axis=1), aspect="auto", vmax=3, vmin=-3)
            ax.set_yticks(np.arange(len(columns)))
            ax.set_yticklabels(columns)
            cbar = fig.colorbar(cax, ax=ax)
            cbar.set_label("strength")
            plt.show()"""

            # Inspect differences in features between 4 and 8 item WM
            features_long = zscore(np.array(features_long), axis=0)
            #features_long = np.array(features_long)
            features_long[features_long > 3] = 3
            features_long[features_long < -3] = -3
            set_size = set_size[:100]
            features_long = features_long[:100]

            # Select only 4 and 8
            idx_4 = np.where(set_size == 4)[0]
            idx_8 = np.where(set_size == 8)[0]

            features_4 = features_long[idx_4]
            features_8 = features_long[idx_8]

            mean_4 = features_4.mean(axis=0)
            sem_4 = features_4.std(axis=0) / np.sqrt(len(idx_4))

            mean_8 = features_8.mean(axis=0)
            sem_8 = features_8.std(axis=0) / np.sqrt(len(idx_8))

            # T-test
            p_vals = np.array([ttest_ind(features_4[:, i], features_8[:, i]).pvalue for i in range(features_4.shape[1])])

            # Plot
            x = np.arange(len(columns))
            width = 0.35

            fig, ax = plt.subplots(figsize=(12, 6))
            bar1 = ax.bar(x - width/2, mean_4, width, yerr=sem_4, label='Set Size 4', capsize=4)
            bar2 = ax.bar(x + width/2, mean_8, width, yerr=sem_8, label='Set Size 8', capsize=4)

            # Mark significant differences
            for i, p in enumerate(p_vals):
                if p < 0.05:
                    print(p)
                    ax.text(i, max(mean_4[i], mean_8[i]) + 0.1, '*', ha='center', va='bottom', fontsize=12, color='red')

            ax.set_ylabel('Normalized Feature Value')
            ax.set_title(f'Subject {subject}, Channel {ch} - Set Size Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(columns, rotation=90)
            ax.legend()
            plt.tight_layout()

            if (p_vals < 0.05).sum() < 1:
                plt.close()
plt.show()
print("h")

# compare entries
"""idx_4 = np.where(set_size == 4)[0]
idx_6 = np.where(set_size == 6)[0]
idx_8 = np.where(set_size == 8)[0]
p_values = compute_p_tfr(features_long, idx_4, idx_8)

fig, axes = plt.subplots(1, 3, figsize=(12, 5))
cax = axes[0].imshow(p_values.T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=0, vmax=0.05)
cbar = fig.colorbar(cax, ax=axes[0])
cbar.set_label("P-value")
axes[0].set_yticks(np.arange(len(columns)))
axes[0].set_yticklabels(columns)
plt.title(f"Subject {subject} Channel {ch_name}")

# Add average features values for 4 and 8
mean_4 = zscore(np.mean(features_long[idx_4], axis=0), axis=0)
mean_8 = zscore(np.mean(features_long[idx_8], axis=0), axis=0)
axes[1].imshow(mean_4.T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
axes[2].imshow(mean_8.T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
plt.subplots_adjust(left=0.3)

plt.show()
#fig.savefig(f"..\\figures\\tfr_subject_0{subject}_{ch}_stats_4_8_features_percent.png")

plt.close()"""


