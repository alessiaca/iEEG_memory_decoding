# Run decoding on BIH server with parallel processing

import py_neuromodulation as nm
import numpy as np
import os
import mne
from catboost import CatBoostClassifier
from sklearn import metrics, model_selection, linear_model, svm
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import matplotlib
import matplotlib.pyplot as plt
import sqlite3
import sys
from scipy.stats import zscore
from joblib import Parallel, delayed
from itertools import product
import random
seed = 42

def run_cross_val(X, y, n_folds=5):

    """model = CatBoostClassifier(iterations=200,
                                      depth=4,
                                      learning_rate=0.02,
                                      l2_leaf_reg=20,
                                      random_strength=5,
                                      verbose=False
                                      )"""
    model = LogisticRegression()

    pipeline = Pipeline([
    ('scaler', StandardScaler()),      
    ('pca', PCA(n_components=3)),     
    ('clf', model)     
])  
    # Use cross-validation (e.g. stratified K-fold for classification)
    cv = StratifiedKFold(n_splits=7, shuffle=True)

    # Run cross-validation
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    return scores.mean()


def predict_target(chan, referencing, Hz, length):

    subjects = np.arange(1, 10)
    for subject in subjects:

        features = pd.read_csv(f"../../Features/subject_{subject}_{referencing}_Hz_{Hz}_length_{length}/subject_{subject}_{referencing}_Hz_{Hz}_length_{length}_FEATURES.csv")

        # Get all features from channel of interest
        columns = [column for column in features.columns if f'CH{chan-1}_' in column]

        if len(columns) > 0:

            # Load the label
            df = pd.read_csv(f'../../Processed_data/merged/Data_Subject_0{subject}.csv')

            # Set the label
            y = df["Set size"].to_numpy()

            # Reshape into epochs
            features_long = []
            for i in range(0, int(np.max(features.time) / 6000)):
                features_epoch = features[(features.time < 6000*(i+1)) & (features.time > 6000*i+1000)]
                # Select only the columns of interest
                features_epoch = features_epoch[columns].to_numpy()
                # Normalize by calculating percent in respect to first entry, keep only the last 2 seconds
                features_epoch_norm = (features_epoch[int(-3 * Hz):, :] - features_epoch[0, :]) / features_epoch[0, :]
                features_long.append(features_epoch_norm)

            # Normalize
            #features_long = zscore(np.array(features_long), axis=0)
            features_long = np.array(features_long)
            #features_long[features_long > 3] = 3
            #features_long[features_long < -3] = -3

            # Reshape the features by concatenating the features from one epoch (but different time points) in one row
            X = np.reshape(features_long, (features_long.shape[0], features_long.shape[1] * features_long.shape[2]))
            
            # Choose only the first 2 sessions (100 samples)
            y = y[:100]
            X = X[:100]

            # Select only 4 and 8
            use_idx_4 = np.where(y == 4)[0][:28]
            use_idx_8 = np.where(y == 8)[0][:28]
            use_idx = np.concatenate((use_idx_4, use_idx_8))
            y = y[use_idx]
            X = X[use_idx]

            # Compute performance of 100 runs (difference due to randomness in umsampling)
            n_runs = 200
            results = []  # list to store rows

            for run in range(n_runs):
                # Run 10-fold cross-validation
                mean_accuracy = run_cross_val(X, y)
            
                # Collect results in a list
                results.append([subject, chan, mean_accuracy, run])
                
            # Run with permuted label 
            n_perms = 200
            for perm in range(n_perms):
                # Run 10-fold cross-validation
                np.random.shuffle(y)
                mean_accuracy = run_cross_val(X, y)
            
                # Collect results in a list
                results.append([subject, chan, mean_accuracy, -perm])
            
            # Create a DataFrame from the collected results
            df = pd.DataFrame(results, columns=['subject', 'channel', 'accuracy', 'run'])
            
            # Connect to SQLite database once
            conn = sqlite3.connect(f"../../Decoding_results/{referencing}_Hz_{Hz}_length_{length}/subject_{subject}_{referencing}_Hz_{Hz}_length_{length}.db")
            
            # Append the entire dataframe to the database
            df.to_sql("selected_data", conn, if_exists="append", index=False)
            
            # Commit and close the connection
            conn.commit()
            conn.close()


referencing = "bip"
Hz = 5
length = 1000
channels = np.arange(1, 65)
os.makedirs(f"../../Decoding_results/{referencing}_Hz_{Hz}_length_{length}", exist_ok=True)
Parallel(n_jobs=-1)(
    delayed(predict_target)(i, referencing, Hz, length) for i in channels
)
#predict_target(1, referencing, Hz, length)
