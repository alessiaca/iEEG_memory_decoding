# Load r-maps
# Transform in nifti files

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr
from scipy.io import loadmat
import matplotlib.pyplot as plt
import nibabel as nib
n = 385

# Save the correlation map for further analysis
r = loadmat(f"../data_epochs/metadata/r_maps_2_9/Channel_{n}_func_seed_AvgR_Fz.mat")["corr"]
plt.hist(r.flatten())
plt.show()
p = loadmat(f"../data_epochs/metadata/r_maps_2_9/Channel_{n}_func_seed_AvgR_Fz.mat")["p"]
plt.hist(p.flatten())
plt.show()

# Create the r map
path = f"../data_epochs/metadata/nifti/fmaps_gsp/Subject_01_bipolar_1_func_seed_AvgR_Fz.nii"
nii = nib.load(path)
nii_new = nib.Nifti1Image(r, nii.affine, nii.header)
nib.save(nii_new, f"../data_epochs/metadata/nifti/channels_all.nii")


"""for sub in range(9):
    # Save the correlation map for further analysis
    r = loadmat(f"../data_epochs/metadata/r_maps_2/Subject_{sub+1}_func_seed_AvgR_Fz.mat")["corr"]
    plt.hist(r.flatten())
    plt.show()
    p = loadmat(f"../data_epochs/metadata/r_maps_2/Subject_{sub+1}_func_seed_AvgR_Fz.mat")["p"]
    plt.hist(p.flatten())
    plt.show()

    # Create the r map
    path = f"../data_epochs/metadata/nifti/fmaps_gsp/Subject_01_bipolar_1_func_seed_AvgR_Fz.nii"
    nii = nib.load(path)
    nii_new = nib.Nifti1Image(r, nii.affine, nii.header)
    nib.save(nii_new, f"../data_epochs/metadata/nifti/r_map_subject_{sub}_2.nii")"""

