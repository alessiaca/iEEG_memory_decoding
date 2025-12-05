# Extract features for each subject

import py_neuromodulation as nm
import numpy as np
import pandas as pd
from scipy.stats import zscore, ttest_ind
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

referencing = "car_all"
length = 800
Hz = 5

subjects = np.arange(1, 10)

for subject in subjects:

    # Load the features
    features = pd.read_csv(f"features/subject_{subject}_{referencing}_Hz_{Hz}_length_{length}/subject_{subject}_{referencing}_Hz_{Hz}_length_{length}_FEATURES.csv")

    # Load the label (set size)
    df = pd.read_csv(f'../../../Processed_data/merged/Data_Subject_0{subject}.csv')
    set_size = df["Set size"].to_numpy()

    # Build matrix for easy handling
    features_long_sub = []

    # Loop over channels
    for ch in range(64):
        ch_name = f'CH{ch}_'
        columns = [column for column in features.columns if ch_name in column]
        features_long_ch = []

        if len(columns) > 0:

            for i in range(0, int(np.max(features.time) / 6000)):
                features_epoch = features[(features.time <= 6000*(i+1)+1) & (features.time > 6000*i+800)]
                # Select only the columns of interest
                features_epoch_np = features_epoch[columns].to_numpy()
                features_long_ch.append(features_epoch_np)

            features_long_sub.append(features_long_ch)

    features_long = np.moveaxis(np.array(features_long_sub), 0, 1)

    # Save features
    np.save(f"features/Subject_{subject}_{referencing}_Hz_{Hz}_length_{length}.npy", features_long)
