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
from sklearn import metrics, model_selection, linear_model, svm
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
np.random.seed(420)

subjects = np.arange(1, 10)
ratio = []

for subject in subjects[:]:

    feature_reader = nm.analysis.FeatureReader(
                    feature_dir="features",
                    feature_file=f"subject_{subject}"
                )

    # Load the label
    df = pd.read_csv(f'..\data_epochs\merged\Data_Subject_0{subject}.csv')

    # Set the label
    feature_reader.label_name = "workload"
    feature_reader.label = df["Set size"].to_numpy()

    # Reshape the feature array if needed
    reshaped_data = []
    new_columns = []

    for col in feature_reader.feature_arr:
        col_values = feature_reader.feature_arr[col].values  # Extract column as numpy array
        new_cols = np.vstack([col_values[::3], col_values[1::3], col_values[2::3]]).T#, col_values[3::6], col_values[4::6]]).T
        reshaped_data.append(new_cols)

        # Generate new column names
        new_columns.extend([f"{col}_1", f"{col}_2", f"{col}_3"])#, f"{col}_4", f"{col}_5"])

    # Convert list of arrays into a DataFrame
    reshaped_df = pd.DataFrame(np.hstack(reshaped_data), columns=new_columns)
    feature_reader.feature_arr = reshaped_df

    model = linear_model.LogisticRegression()
    #model = svm.SVC(kernel="linear")

    # Select only 4 and 8 set sizes
    use_idx = np.isin(feature_reader.label, [4, 8])
    feature_reader.label = feature_reader.label[use_idx]
    feature_reader.feature_arr = feature_reader.feature_arr[use_idx]

    ratio.append(np.sum(feature_reader.label == 4) / np.sum(feature_reader.label == 8))

    # Balance the groups
    n_min = np.min([np.sum(feature_reader.label == 4), np.sum(feature_reader.label == 8)])
    # Randomly sample the minimum number from both groups
    idx = np.concatenate((np.random.choice(np.where(feature_reader.label == 4)[0], size=n_min, replace=False),
                          np.random.choice(np.where(feature_reader.label == 8)[0], size=n_min, replace=False)))
    use_idx = [True if i in idx else False for i in range(len(feature_reader.label))]
    feature_reader.label = feature_reader.label[use_idx]
    feature_reader.feature_arr = feature_reader.feature_arr[use_idx]

    feature_reader.decoder = nm.analysis.Decoder(
        features=feature_reader.feature_arr,
        label=feature_reader.label,
        label_name=feature_reader.label_name,
        used_chs=feature_reader.used_chs,
        STACK_FEATURES_N_SAMPLES=False,
        eval_method=metrics.accuracy_score,
        model=model,
        cv_method=model_selection.KFold(n_splits=10, shuffle=False),
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
    axes[0].axhline(1/2, color="red")
    axes[0].set_ylim([0, 0.9])
    axes[0].set_title("train")

    axes[1].bar(np.arange(len(df_per)), df_per["performance_test"])
    axes[1].axhline(1 / 2, color="red")
    axes[1].set_ylim([0, 0.9])
    axes[1].set_title("test")
    plt.show()

print(ratio)