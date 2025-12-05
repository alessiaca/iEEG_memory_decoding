# R_map validation across subjects and channels

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.formula.api as smf
import os

# Parameters
referencing = "bip"
Hz = 10
length = 1000
n_runs = 100
measure = "accuracy"
remove_below_chance = False
mode = "Functional"
flip_mode = "all"
model_name = f"{referencing}_Hz_{Hz}_length_{length}_runs_{n_runs}"
folder_path = f"../../Maps/Correlation/{model_name}/flip_{flip_mode}/{mode}/across_subjects_channels"
p_thres = 0.05
subjects = np.arange(1, 10)

# Load decoding results
df_decoding = pd.read_csv(f"../../Decoding_results/{model_name}.csv")

if remove_below_chance:
    df_decoding = df_decoding[df_decoding["above_chance"]]

#df_decoding = df_decoding[df_decoding[measure] > 0.5]
#df_decoding = df_decoding[df_decoding[measure] < 0.75]

df_decoding["similarity"] = np.nan  # Add similarity column

p_map_0 = loadmat(f"{folder_path}/without_subject_0_channel_0_func_seed_AvgR_Fz.mat")["p"]
p_maps = []
for idx, row in df_decoding.iterrows():
    subject = row["subject"]
    channel = row["channel"]

    # Load functional connectivity map
    path = None
    flip_options = ["true", "false"] if flip_mode == "all" else [flip_mode]
    try:
        for flipped in flip_options:
            path = f"../../Maps/Functional_connectivity/Subject_{subject}_bipolar_{channel}_flipped_{flipped}_func_seed_AvgR_Fz.nii"
            if os.path.exists(path):
                break
        image = nib.load(path).get_fdata()

        # Load r-map calculated from remaining channels and subjects
        r_map = loadmat(f"{folder_path}/without_subject_{subject}_channel_{channel}_func_seed_AvgR_Fz.mat")["corr"]
        p_map = loadmat(f"{folder_path}/without_subject_{subject}_channel_{channel}_func_seed_AvgR_Fz.mat")["p"]
        p_maps.append(p_map)

        valid = ~np.isnan(image.flatten()) & ~np.isnan(r_map.flatten()) & (p_map_0.flatten() < p_thres)
        similarity, _ = pearsonr(image.flatten()[valid], r_map.flatten()[valid])

        df_decoding.at[idx, "similarity"] = similarity

    except Exception as e:
        pass

# Check p maps
print((np.array(p_maps) < 0.05).all(axis=0).sum()/(p_map_0 < 0.05).sum())
print((p_map_0 < 0.05).sum())

# Drop rows without similarity values
df_decoding = df_decoding.dropna(subset=["similarity"])

# Plot results
plt.figure(figsize=(10, 6))
sb.regplot(
    x=measure, y="similarity", data=df_decoding,
    scatter_kws={'s': 30, 'alpha': 0.6},
    line_kws={"linewidth": 1.5},
    color="magenta"
)
r, p = spearmanr(df_decoding[measure], df_decoding["similarity"])
plt.title(f"Regression across all subjects and channels r = {r:.3f}, p = {p:.3f} \n Remove channels below chance {remove_below_chance}; map masked with p value below {p_thres}", fontsize=12)
plt.xlabel("Decoding Performance", fontsize=12)
plt.ylabel("Functional Connectivity Similarity", fontsize=12)
plt.tight_layout()

# Save plot
plt.savefig(f"../../Figures/across_subjects_channels_{remove_below_chance}_{p_thres}_validation.png", dpi=300)
plt.show()

