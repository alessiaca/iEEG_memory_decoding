# Calculate r maps

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr
from scipy.io import savemat

# Define parameters
referencing = "bip"
Hz = 10
length = 1000
n_permutation = 1000
model_name = f"{referencing}_Hz_{Hz}_length_{length}_perm_{n_permutation}"
measure = "performance"  # or p
threshold = 0.55 # or higher value
subjects = np.arange(1, 10)

# Load the decdoing results
df_decoding = pd.read_csv(f"../Decoding_results/{model_name}.csv")

# Filter the results to use only the one of interest
df_decoding = df_decoding[(df_decoding[measure] > threshold) & (df_decoding["subject"].isin(subjects))]

# Load the functional connectivity maps for each subject/channel combination
image_all = []
# Loop over rows in the DataFrame and extract subject and channel
for _, row in df_decoding.iterrows():
    subject = int(row["subject"])  
    channel = int(row["channel"]) + 1
    path = f"../Maps/Functional_connectivity/Subject_0{subject}_bipolar_{channel}_func_seed_AvgR_Fz.nii"
    image = nib.load(path).get_fdata()
    image_all.append(image)

image_all = np.array(image_all)

# Extract the measure
decoding_measure = df_decoding[measure].to_numpy()

for i in range(len(subjects)+1):

    if i == 0:
        idx_train = np.arange(len(decoding_measure))
    else:
        # Delete the test subject from the training set
        idx_train = df_decoding[df_decoding["subject"] != i].index

    # Calculate the correlation map over the whole set
    corr = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    p = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            for l in range(image.shape[2]):
                corr[j, k, l], p[j, k, l] = spearmanr(image_all[idx_train, j, k, l], decoding_measure[idx_train])


    # Save the correlation map for further analysis
    savemat(f"../data_epochs/metadata/corr_maps/Channel_{ch}_func_seed_AvgR_Fz.mat",
            {"corr":corr, "p":p})
    
    # Save as nifti file
    path = f"../data_epochs/metadata/nifti/fmaps_gsp/Subject_01_bipolar_1_func_seed_AvgR_Fz.nii"
    nii = nib.load(path)
    nii_new = nib.Nifti1Image(r, nii.affine, nii.header)
    nib.save(nii_new, f"../data_epochs/metadata/nifti/channels_all.nii")

