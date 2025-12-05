import numpy as np
import pandas as pd

subjects = np.arange(1, 10)
for subject in subjects:

    # Load the label
    df = pd.read_csv(f'../../Processed_data/merged/Data_Subject_0{subject}.csv')

    # Set the label
    y = df["Set size"].to_numpy()

    # Print minimum number of same entries
    unique_counts = np.unique(y, return_counts=True)
    print(f'Subject {subject}: Minimum number of same entries = {np.min(unique_counts[1])}')
