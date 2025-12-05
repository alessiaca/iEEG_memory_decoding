# R_map validation within subjects

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sb
import os

# Parameters
referencing = "bip"
Hz = 10
length = 1000
n_runs = 100
measure = "accuracy"
remove_below_chance = False
flip_mode = "false"
model_name = f"{referencing}_Hz_{Hz}_length_{length}_runs_{n_runs}"
p_thres = 0.05
subjects = np.arange(1, 10)

# Load decoding results
df_decoding = pd.read_csv(f"../../Decoding_results/{model_name}.csv")

if remove_below_chance:
    df_decoding = df_decoding[df_decoding["above_chance"]]

df_decoding["similarity"] = np.nan  # Add similarity column

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
colors = sb.color_palette("husl", len(subjects))
for i, subject in enumerate(subjects):

    df_decoding_sub = df_decoding[df_decoding["subject"] == subject]
    folder_path = f"../../Maps/Correlation/{model_name}/flip_{flip_mode}/within_subjects/subject_{subject}"

    p_maps = []

    for idx, row in df_decoding_sub.iterrows():
        channel = int(row["channel"])

        try:
            # Load the functional connectivity of each channel
            path = None
            flip_options = ["true", "false"] if flip_mode == "all" else [flip_mode]
            for flipped in flip_options:
                path = f"../../Maps/Functional_connectivity/Subject_{subject}_bipolar_{channel}_flipped_{flipped}_func_seed_AvgR_Fz.nii"
                if os.path.exists(path):
                    break
            image = nib.load(path).get_fdata()

            # Load the r-map computed without the channel
            r_map = loadmat(f"{folder_path}/without_{channel}_func_seed_AvgR_Fz.mat")["corr"]
            #p_map = loadmat(f"{folder_path}/without_{channel}_func_seed_AvgR_Fz.mat")["p"]
            #r_map = loadmat(f"{folder_path}/without_0_func_seed_AvgR_Fz.mat")["corr"]
            p_map = loadmat(f"{folder_path}/without_0_func_seed_AvgR_Fz.mat")["p"]
            #print((p_map.flatten() < p_thres).sum())
            p_maps.append(p_map)

            #valid = ~np.isnan(image.flatten()) & ~np.isnan(r_map.flatten())
            valid = ~np.isnan(image.flatten()) & ~np.isnan(r_map.flatten()) & (p_map.flatten() < p_thres)
            similarity, _ = pearsonr(image.flatten()[valid], r_map.flatten()[valid])

            df_decoding_sub.at[idx, "similarity"] = similarity

        except Exception as e:
            #print(e)
            pass
    #print((np.array(p_maps).mean(axis=0) < 0.05).sum())
    p_map = loadmat(f"{folder_path}/without_0_func_seed_AvgR_Fz.mat")["p"]
    print((np.array(p_maps) < 0.05).all(axis=0).sum()/(p_map < 0.05).sum())
    print((p_map < 0.05).sum())

    df_decoding_sub = df_decoding_sub.dropna(subset=["similarity"])

    # Plot results
    ax = axes[i//3, i%3]
    sb.regplot(
        x=measure, y="similarity", data=df_decoding_sub,
        color=colors[i],
        scatter_kws={'s': 30, 'alpha': 0.6},
        line_kws={"linewidth": 1.5},
        ax=ax
    )
    r, p = spearmanr(df_decoding_sub[measure], df_decoding_sub["similarity"])
    ax.set_title(f"Subject {subject}, r = {r:.3f}, p = {p:.3f}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Decoding Performance", fontsize=9)
    ax.set_ylabel("Functional Connectivity \nSimilarity", fontsize=9)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

plt.suptitle(f"Remove channels below chance {remove_below_chance}; map masked with p value below {p_thres}")
plt.subplots_adjust(wspace=0.5, hspace=0.5)

# Save plot
plt.savefig(f"../../Figures/within_subjects_{remove_below_chance}_{p_thres}_validation.png", dpi=300)
plt.show()
