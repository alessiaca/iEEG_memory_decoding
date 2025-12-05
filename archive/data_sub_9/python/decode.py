# Run decoding on BIH server with parallel processing

import py_neuromodulation as nm
import numpy as np
import os
import mne
from catboost import CatBoostClassifier
from sklearn import metrics, model_selection, linear_model, svm
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
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

    # Set up 10-fold cross-validation
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    cv_accuracies_test = []
    cv_accuracies_train = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Find unique class labels and their counts
        """classes, counts = np.unique(y_train, return_counts=True)

        # Identify majority and minority class
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        
        # Extract minority samples
        X_train_minority = X_train[y_train == minority_class]
        y_train_minority = y_train[y_train == minority_class]
        
        # Compute number of samples needed to balance classes
        num_samples_to_add = max(counts) - min(counts)
        
        # Randomly duplicate minority samples
        indices = np.random.choice(len(X_train_minority), size=num_samples_to_add, replace=False)
        
        # Oversample
        X_train_oversampled = np.vstack((X_train, X_train_minority[indices]))
        y_train_oversampled = np.hstack((y_train, y_train_minority[indices]))
        
        # Update training data
        X_train, y_train = X_train_oversampled, y_train_oversampled"""

        # Initialize CatBoost model with basic settings
        """model = CatBoostClassifier(iterations=200,
                                      depth=4,
                                      learning_rate=0.02,
                                      l2_leaf_reg=20,
                                      random_strength=5,
                                      verbose=False
                                      )

        # Train CatBoost model
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=20)"""
        
        model = LogisticRegression(class_weight='balanced')
        #model = DummyClassifier(strategy='stratified')
        model.fit(X_train, y_train)

        # Make predictions and calculate accuracy
        y_pred = model.predict(X_test)
        #y_pred_inverted = np.where(y_pred == 4, 8, 4)
        accuracy_test = balanced_accuracy_score(y_test, y_pred)
        cv_accuracies_test.append(accuracy_test)

        y_pred_train = model.predict(X_train)
        accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
        cv_accuracies_train.append(accuracy_train)

    # Average accuracy across all folds
    mean_accuracy_test = np.mean(cv_accuracies_test)
    mean_accuracy_train = np.mean(cv_accuracies_train)

    return mean_accuracy_test


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
                features_epoch_norm = (features_epoch[int(-1.5 * Hz):, :] - features_epoch[0, :]) / features_epoch[0, :]
                features_long.append(features_epoch_norm.mean(axis=0))

            # Normalize
            #features_long = zscore(np.array(features_long), axis=0)
            features_long = np.array(features_long)
            #features_long[features_long > 3] = 3
            #features_long[features_long < -3] = -3

            # Reshape the features by concatenating the features from one epoch (but different time points) in one row
            X = features_long #np.reshape(features_long, (features_long.shape[0], features_long.shape[1] * features_long.shape[2]))
            
            # Choose only the first 2 sessions (100 samples)
            y = y[:100]
            X = X[:100]

            # Select only 4 and 8
            use_idx = np.isin(y, [4, 8])
            y = y[use_idx]
            X = X[use_idx]

            # Compute performance of 100 runs (difference due to randomness in umsampling)
            n_runs = 100
            results = []  # list to store rows

            for run in range(n_runs):
                # Run 10-fold cross-validation
                mean_accuracy = run_cross_val(X, y)
            
                # Collect results in a list
                results.append([subject, chan, mean_accuracy, run])
                
            # Run with permuted label 
            n_perms = 100
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


referencing = "none"
Hz = 5
length = 1000
channels = np.arange(1, 65)
os.makedirs(f"../../Decoding_results/{referencing}_Hz_{Hz}_length_{length}", exist_ok=True)
Parallel(n_jobs=1)(
    delayed(predict_target)(i, referencing, Hz, length) for i in channels
)
#predict_target(1, referencing, Hz, length)
