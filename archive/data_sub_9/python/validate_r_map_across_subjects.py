# R_map validation across subjects 

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.formula.api as smf
import os

# todo: resample to same dimensions (SPM?)
# generate fmaps for original electrode location
# inspect features 
# inspect unreferenced data
# train model on all the data 

# Parameters
referencing = "none"
Hz = 5
length = 1000
n_runs = 200
measure = "accuracy"
mode = "Functional"
flip_mode = "false"
model_name = f"{referencing}_Hz_{Hz}_length_{length}_runs_{n_runs}"
folder_path = f"../../Maps/Correlation/{model_name}/flip_{flip_mode}/{mode}/across_subjects_all"
p_thres = 0.05
subjects = np.arange(1, 10)

# Load the ROI
#path = "../../Atlas/rc1mask.nii"
path = "../../Atlas/rFornix.nii"
fornix_image = nib.load(path).get_fdata()

# Load decoding results
df_decoding = pd.read_csv(f"../../Decoding_results/{model_name}.csv")

# Load the overlap with the grey matter
df_overlap = pd.read_csv(f"../../Processed_data/metadata/grey_matter_overlap.csv")

# Filter the channels which have above-chance level decoding for all runs (with random seeds)
df_decoding = df_decoding[(df_decoding[measure] > 0.5)]

# Filter out the subjects of interest
df_decoding = df_decoding[df_decoding["subject"].isin(subjects)]

# Filter out the channels that overlap with the grey matter at least 25 %
#df_decoding = df_decoding[(df_overlap["percent_overlap"] > 0.1)]

df_decoding["similarity"] = np.nan  # Add similarity column

for i in subjects:
    r_map = loadmat(f"{folder_path}/without_{i}_func_seed_AvgR_Fz.mat")["corr"]
    p_map = loadmat(f"{folder_path}/without_{i}_func_seed_AvgR_Fz.mat")["p"]
    df_sub = df_decoding[df_decoding["subject"] == i]

    for idx, row in df_sub.iterrows():
        subject = int(row["subject"])
        channel = int(row["channel"])

        path = None
        flip_options = ["true", "false"] if flip_mode == "true" else ["false"]
        try:
            for flipped in flip_options:
                if mode == "Functional":
                    path = f"../../Maps/Functional_connectivity/Subject_{subject}_{referencing}_{channel}_flipped_{flipped}_func_seed_AvgR_Fz.nii"
                else:
                    path = f"../../Maps/Structural_connectivity/Subject_{subject}_{referencing}_{channel}_struc_seed.nii"
                if os.path.exists(path):
                    break
            image = nib.load(path).get_fdata()
            valid = ~np.isnan(image.flatten()) & ~np.isnan(r_map.flatten()) &(fornix_image.flatten() > 0.1)# (p_map.flatten() < p_thres) #(fornix_image.flatten() > 0.1) # 
            similarity, _ = spearmanr(image.flatten()[valid], r_map.flatten()[valid])

            df_decoding.at[idx, "similarity"] = similarity
        except:
            pass

# Drop rows without similarity values
df_lmm = df_decoding.dropna(subset=["similarity"])

# Linear Mixed Model
model = smf.mixedlm(f"{measure} ~ similarity", df_lmm, groups=df_lmm["subject"])
result = model.fit()
print(result.summary())

# Plot results
plt.figure(figsize=(10, 6))
subjects = df_lmm["subject"].unique()
colors = sb.color_palette("husl", len(subjects))
for i, subj in enumerate(subjects):
    df_sub = df_lmm[df_lmm["subject"] == subj]
    sb.regplot(
        x=measure, y="similarity", data=df_sub,
        scatter_kws={'s': 30, 'alpha': 0.6},
        line_kws={"linewidth": 1.5},
        color=colors[i],
        label=f"Subject {subj}"
    )
r, p = spearmanr(df_lmm[measure], df_lmm["similarity"])
plt.title(f"Per-subject regressions Group r = {r:.3f}, p = {p:.3f} \n {mode}; map masked with p value below {p_thres}", fontsize=12)
plt.xlabel("Decoding Performance", fontsize=12)
plt.ylabel(f"{mode} Connectivity Similarity", fontsize=12)
plt.legend(title="Subjects", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()

# Save plot
plt.savefig(f"../../Figures/{mode}_flipped_{flip_mode}_{p_thres}_validation.png", dpi=300)
plt.show()
