# Predict target (e.g. workload) from iEEG features during the maintanance period

# TODO
# Try different re-reference methods
# Try feature normalization
# Try different ML models
# Other features
# Time resolved
# Balance groups

import py_neuromodulation as nm
import numpy as np
import os
import mne
from sklearn import metrics, model_selection, linear_model
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

subjects = np.arange(1, 10)
files = os.listdir("..\data_epochs")

for subject in subjects:

    feature_reader = nm.analysis.FeatureReader(
                    feature_dir="features",
                    feature_file=f"subject_{subject}"
                )

    # Load the label
    df = pd.read_csv(f'..\data_epochs\Data_Subject_0{subject}.csv')

    # Set the label
    feature_reader.label_name = "workload"
    feature_reader.label = df["Set size"].to_numpy()

    model = linear_model.LogisticRegression()

    feature_reader.decoder = nm.analysis.Decoder(
        features=feature_reader.feature_arr,
        label=feature_reader.label,
        label_name=feature_reader.label_name,
        used_chs=feature_reader.used_chs,
        STACK_FEATURES_N_SAMPLES=False,
        eval_method=metrics.accuracy_score,
        model=model,
        cv_method=model_selection.KFold(n_splits=3, shuffle=True),
        VERBOSE=True
    )

    performances = feature_reader.run_ML_model(
        estimate_channels=True,
        estimate_gridpoints=False,
        estimate_all_channels_combined=False,
        save_results=True,
        )

    df_per = feature_reader.get_dataframe_performances(performances)
    # Plot the performances
    fig, axes = plt.subplots(1, 2)
    axes[0].bar(np.arange(len(df_per)), df_per["performance_train"])
    axes[0].axhline(1/3, color="red")
    axes[0].set_ylim([0, 0.6])
    axes[0].set_title("train")

    axes[1].bar(np.arange(len(df_per)), df_per["performance_test"])
    axes[1].axhline(1 / 3, color="red")
    axes[1].set_ylim([0, 0.6])
    axes[1].set_title("test")
    plt.show()