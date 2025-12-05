# Calculate r maps

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr
from scipy.io import savemat
import os

# Define parameters
referencing = "bip"
Hz = 10
length = 1000
n_permutation = 501
model_name = f"{referencing}_Hz_{Hz}_length_{length}_perm_{n_permutation}"
measure = "performance"  # or p
threshold = 0 # or higher value
subjects = np.arange(1, 10)

# Load the decoding results
df_decoding = pd.read_csv(f"../../Decoding_results/{model_name}.csv")

# Filter the results to use only the one of interest
df_decoding = df_decoding[(df_decoding[measure] > threshold) & (df_decoding["subject"].isin(subjects))]

# Load the functional connectivity maps for each subject/channel combination
image_all = []
# Loop over rows in the DataFrame and extract subject and channel
for _, row in df_decoding.iterrows():
    subject = int(row["subject"])  
    channel = int(row["channel"])
    path = f"../../Maps/Functional_connectivity/Subject_{subject}_bipolar_{channel}_func_seed_AvgR_Fz.nii"
    image = nib.load(path).get_fdata()
    image_all.append(image)

image_all = np.array(image_all)

# Extract the measure
decoding_measure = df_decoding[measure].to_numpy()

for i in np.hstack(([0], subjects)):

    if i == 0:
        idx_train = np.arange(len(decoding_measure))
    else:
        # Delete the test subject from the training set
        idx_train = np.where(df_decoding["subject"] != i)[0]

    # Calculate the correlation map over the whole set
    corr = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    p = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            for l in range(image.shape[2]):
                corr[j, k, l], p[j, k, l] = spearmanr(image_all[idx_train, j, k, l], decoding_measure[idx_train])

    # Save the correlation map for further analysis
    folder_path = f"../../Maps/Correlation/{model_name}/measure_{measure}_threshold_{threshold}"
    os.makedirs(folder_path, exist_ok=True)
    savemat(f"{folder_path}/without_{i}_func_seed_AvgR_Fz.mat",
            {"corr":corr, "p":p})
    
    # Save as nifti file
    nii = nib.load(path)
    nii_new = nib.Nifti1Image(corr, nii.affine, nii.header)
    nib.save(nii_new, f"{folder_path}/{threshold}_without_{i}_func_seed_AvgR_Fz.nii")

