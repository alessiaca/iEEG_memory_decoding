# Run decoding on BIH server with parallel processing

import py_neuromodulation as nm
import numpy as np
import os
import mne
from catboost import CatBoostClassifier
from sklearn import metrics, model_selection, linear_model, svm
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sqlite3
import sys
from joblib import Parallel, delayed
from itertools import product
matplotlib.use("TkAgg")
np.random.seed(420)


def predict_target(subject, n_channel):

    feature_reader = nm.analysis.FeatureReader(
                    feature_dir="features",
                    feature_file=f"subject_{subject}"
                )

    if len(feature_reader.channels) > n_channel:

        channel = feature_reader.ch_names[n_channel]

        # Load the label
        df = pd.read_csv(f'..\data_epochs\merged\Data_Subject_0{subject}.csv')

        # Set the label
        feature_reader.label_name = "workload"
        feature_reader.label = df["Set size"].to_numpy()

        model = linear_model.LogisticRegression()

        # Select only 4 and 8 set sizes
        use_idx = np.isin(feature_reader.label, [4, 8])
        feature_reader.label = feature_reader.label[use_idx]
        feature_reader.feature_arr = feature_reader.feature_arr[use_idx]

        feature_reader.decoder = nm.analysis.Decoder(
            features=feature_reader.feature_arr,
            label=feature_reader.label,
            label_name=feature_reader.label_name,
            used_chs=[channel],
            STACK_FEATURES_N_SAMPLES=False,
            eval_method=metrics.accuracy_score,
            undersampling=False,
            model=model,
            cv_method=model_selection.KFold(n_splits=10, shuffle=True),
            VERBOSE=False
        )

        performances = feature_reader.run_ML_model(
            estimate_channels=True,
            estimate_gridpoints=False,
            estimate_all_channels_combined=False,
            save_results=True,
            output_name=f"Chan_{n_channel}"
            )
        df_per = feature_reader.get_dataframe_performances(performances)
        df_per = df_per[["performance_test", "performance_train", "ch"]]
        df_per["permutation"] = -1

        # Compute performance of permutation
        n_permutations = 5
        for perm in range(n_permutations):
            print(perm)
            feature_reader.label = np.random.permutation(feature_reader.label)
            feature_reader.decoder = nm.analysis.Decoder(
                features=feature_reader.feature_arr,
                label=feature_reader.label,
                label_name=feature_reader.label_name,
                used_chs=[channel],
                STACK_FEATURES_N_SAMPLES=False,
                eval_method=metrics.accuracy_score,
                undersampling=False,
                model=model,
                cv_method=model_selection.KFold(n_splits=10, shuffle=True),
                VERBOSE=True
            )

            performances_perm = feature_reader.run_ML_model(
                estimate_channels=True,
                estimate_gridpoints=False,
                estimate_all_channels_combined=False,
                save_results=True,
                output_name=f"Chan_{n_channel}"
            )
            # Extract performance
            df_per_perm = feature_reader.get_dataframe_performances(performances_perm)
            df_per_perm = df_per_perm[["performance_test", "performance_train", "ch"]]
            df_per_perm["permutation"] = perm
            # Append to dataframe
            df_per = pd.concat([df_per, df_per_perm], ignore_index=True)

        # Connect to SQLite database (create or open the database)
        conn = sqlite3.connect(f"subject_{subject}.db")

        # Append the dataframe to the database
        df_per.to_sql("selected_data", conn, if_exists="append", index=False)

        # Commit the transaction and close the connection
        conn.commit()
        conn.close()

if __name__ == "__main__":

    subjects = np.arange(1, 10)
    n_channels = 64
    #input1 = int(sys.argv[1]) - 1
    output = list(product(range(len(subjects)), range(n_channels)))

    #predict_target(subjects[output[0][0]-1], output[0][1])
    Parallel(n_jobs=2)(delayed(predict_target)(1, x) for x in range(5))

    # Save test as csv
    conn = sqlite3.connect(f"subject_1.db")
    df = pd.read_sql("SELECT * FROM selected_data", conn)
    df.to_csv("test.csv", index=False)
    conn.close()

