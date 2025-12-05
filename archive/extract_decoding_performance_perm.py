# Plot the performance of the decoding
# One overview plot with all subjects displaying the p values
# One plot per subject with the performances

import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use("TkAgg")

# Define parameters
referencing = "bip"
Hz = 10
length = 1000
n_permutation = "linear"
model_name = f"{referencing}_Hz_{Hz}_length_{length}_perm_{n_permutation}"

# Initialize the plots
subjects = np.arange(1, 10)

# Initialize a list to collect data from all subjects
all_subjects_data = []

for subject in subjects:
    db_path = f"../../Decoding_results/{model_name}/subject_{subject}_{referencing}_Hz_{Hz}_length_{length}.db"
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        continue

    df = pd.read_sql("SELECT * FROM selected_data", sqlite3.connect(db_path))
    grouped = df.groupby("channel")
    subject_data = pd.DataFrame({
        "subject": subject,
        "channel": grouped.groups.keys(),
        "performance": grouped.apply(lambda g: g.loc[g["permutation"] == -1, "accuracy"].iloc[-1]),
        "p": grouped.apply(lambda g: np.sum(g.loc[g["permutation"] != -1, "accuracy"] > g.loc[g["permutation"] == -1, "accuracy"].iloc[-1]) / len(g.loc[g["permutation"] != -1, "accuracy"]))
    })
    all_subjects_data.append(subject_data[subject_data["p"] != 0])

combined_data = pd.concat(all_subjects_data, ignore_index=True)
combined_data.to_csv(f"../../Decoding_results/{model_name}.csv", index=False)