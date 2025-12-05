# Calculate r maps

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr
from scipy.io import savemat

subjects = np.arange(1, 10)

# Load all functional connectivity maps and accuracies
image_all = []
accuracy_all = []
for sub in subjects:
    accuracy_sub = pd.read_csv(f"perf_{sub}.csv")
    accuracy_all.append(accuracy_sub.performance)

    channels = np.arange(len(accuracy_sub))
    image_all_subject = []
    for ch in channels:
        path = f"../data_epochs/metadata/nifti/fmaps_gsp/Subject_0{sub}_bipolar_{ch+1}_func_seed_AvgR_Fz.nii"
        image = nib.load(path).get_fdata()
        image_all_subject.append(image)

    image_all_subject = np.array(image_all_subject)
    image_all.append(image_all_subject)


for sub in np.arange(9, len(subjects)+1):

    # Calculate correlations
    print(f"Subject {sub}")

    if sub == len(subjects):
        idx_train = np.arange(len(subjects))
    else:
        # Delete the test subject from the training set
        idx_train = np.delete(np.arange(len(subjects)), sub)

    accuracy_all_tmp = np.array([xs for x in idx_train for xs in accuracy_all[x]])
    image_all_tmp = np.array([xs for x in idx_train for xs in image_all[x]])

    # Delete bad channels
    image_all_tmp = image_all_tmp[accuracy_all_tmp > 0, :, :, :]
    accuracy_all_tmp = accuracy_all_tmp[accuracy_all_tmp > 0]

    # Compute correlation for each voxel
    corr = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    p = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            for l in range(image.shape[2]):
                corr[j, k, l], p[j, k, l] = spearmanr(image_all_tmp[idx_train, j, k, l], accuracy_all_tmp[idx_train])


    # Save the correlation map for further analysis
    savemat(f"../data_epochs/metadata/corr_maps/Subject_{sub}_func_seed_AvgR_Fz.mat",
            {"corr":corr, "p":p})

