# For each channel load the r map which was generated without the data from the channel
# Then compute the distance of the map to the channel
# Correlate the distance with the performance of that channel

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr
from scipy.io import loadmat
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.stats import pearsonr
import seaborn as sb

channels = np.arange(387)
fig, axes = plt.subplots(1, 1, figsize=(5, 5))

for ch in channels:

    # Load the rmap
    r_map = loadmat(f"../data_epochs/metadata/r_maps_2/Channel_{ch}_func_seed_AvgR_Fz.mat")["corr"]

    # Load the performance of each channel
    accuracy_sub = pd.read_csv(f"perf_{sub+1}.csv").performance.to_numpy()

    # Load the functional connectivity maps for each channel
    channels = np.arange(len(accuracy_sub))
    image_sub = []
    for ch in channels:
        path = f"../data_epochs/metadata/nifti/fmaps_gsp/Subject_0{sub+1}_bipolar_{ch + 1}_func_seed_AvgR_Fz.nii"
        image = nib.load(path).get_fdata()
        image_sub.append(image)

    # Delete bad channels
    image_sub = np.array(image_sub)[accuracy_sub > 0.5, :, :, :]
    accuracy_sub = accuracy_sub[accuracy_sub > 0.5]

    # Calculate the distance between the r map and each channel functional map
    similarity_sub = np.zeros(accuracy_sub.shape)
    for ch in np.arange(len(accuracy_sub)):
        image = image_sub[ch, :, :, :]
        valid_indices = ~np.isnan(image.flatten()) & ~np.isnan(r_map.flatten())
        similarity_sub[ch], _ = pearsonr(image.flatten()[valid_indices], r_map.flatten()[valid_indices])

    # Correlate the accuracy and the distance
    r, p = pearsonr(accuracy_sub, similarity_sub)
    print(p)
    p = np.round(p, 3)
    sb.regplot(x=accuracy_sub, y=similarity_sub)
plt.show()


