# Predict target (e.g. workload) from iEEG features during the maintanance period

# TODO
# Try different re-reference methods
# Try feature normalization
# Try different ML models
# Other features
# Time resolved
# Balance groups

# Make bipolar references for iEEG (subsequent, check

import py_neuromodulation as nm
import numpy as np
import os
import mne
from catboost import CatBoostClassifier
from sklearn import metrics, model_selection, linear_model, svm
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
np.random.seed(420)

normalize = True
referencing = "car"
n_split = 1

subjects = np.arange(1, 10)

for subject in subjects:

    feature_reader = nm.analysis.FeatureReader(
                    feature_dir="features",
                    feature_file=f"subject_{subject}_{normalize}_{referencing}_split_{n_split}"
                )

    # Load the label
    df = pd.read_csv(f'..\data_epochs\merged\Data_Subject_0{subject}.csv')

    # Set the label
    feature_reader.label_name = "workload"
    feature_reader.label = df["Set size"].to_numpy()

    model = linear_model.LogisticRegression()
    """model = CatBoostClassifier(iterations=50,
                              depth=4
                              )"""

    # Select only 4 and 8 set sizes
    use_idx = np.isin(feature_reader.label, [4, 8])
    feature_reader.label = feature_reader.label[use_idx]
    feature_reader.feature_arr = feature_reader.feature_arr[use_idx]

    feature_reader.decoder = nm.analysis.Decoder(
        features=feature_reader.feature_arr,
        label=feature_reader.label,
        label_name=feature_reader.label_name,
        used_chs=feature_reader.used_chs,
        STACK_FEATURES_N_SAMPLES=False,
        eval_method=metrics.accuracy_score,
        undersampling=False,
        model=model,
        cv_method=model_selection.KFold(n_splits=10, shuffle=True),
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(np.arange(len(df_per)), df_per["performance_train"])
    axes[0].axhline(1/2, color="red")
    axes[0].set_ylim([0, 0.9])
    axes[0].set_title("train")

    axes[1].bar(np.arange(len(df_per)), df_per["performance_test"])
    axes[1].axhline(1 / 2, color="red")
    axes[1].set_ylim([0, 0.9])
    axes[1].set_title("test")

    # Compute performances in 100 permutations
    n_permutations = 2
    permutation_df_per = []
    for perm in range(n_permutations):
        permuted_label = np.random.permutation(feature_reader.label.copy())
        feature_reader.decoder = nm.analysis.Decoder(
            features=feature_reader.feature_arr,
            label=permuted_label,
            label_name=feature_reader.label_name,
            used_chs=feature_reader.used_chs,
            STACK_FEATURES_N_SAMPLES=False,
            eval_method=metrics.accuracy_score,
            undersampling=False,
            model=model,
            cv_method=model_selection.KFold(n_splits=10, shuffle=True),
            VERBOSE=True
        )

        performances = feature_reader.run_ML_model(
            estimate_channels=True,
            estimate_gridpoints=False,
            estimate_all_channels_combined=False,
            save_results=True,
        )
        permutation_df_per.append(feature_reader.get_dataframe_performances(performances))

    # Compute the p values
    for i, ch in enumerate(feature_reader.used_chs):
        perf_perm_train = [permutation_df_per[x]["performance_train"][permutation_df_per[0]["ch"] == ch] for x in range(len(permutation_df_per))]
        perf_perm_test = [permutation_df_per[x]["performance_test"][permutation_df_per[0]["ch"] == ch] for x in range(len(permutation_df_per))]

        # Plot the performance values from the permuted labels
        axes[0].scatter([i]*n_permutations, perf_perm_train, color="black", s=1)
        axes[1].scatter([i]*n_permutations, perf_perm_test, color="black", s=1)

        # Add th p value
        p_train = sum(perf_perm_train > df_per['performance_train'][i])[0]/n_permutations
        p_test = sum(perf_perm_test > df_per['performance_test'][i])[0]/n_permutations
        for j, p in enumerate([p_train, p_test]):
            if p < 0.05:
                fontweight = "bold"
                color = "red"
            else:
                fontweight = "normal"
                color = "black"
            axes[j].text(i, 0.25, f"{p}", fontsize=6, fontweight=fontweight, color=color, rotation=90, ha="center", va="center")

    plt.show()



