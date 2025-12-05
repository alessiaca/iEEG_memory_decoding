import pandas as pd
import os
import glob

# Set the folder containing the CSV files
folder_path = '../../Fiber_filtering/tracts_overlap_true'  # Change this to your folder path

# Get list of all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Read and concatenate all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Save to a new CSV
output_path = os.path.join(folder_path, 'tracts_overlap_true.csv')
merged_df.to_csv(output_path, index=False)

print(f"Merged {len(csv_files)} files into {output_path}")
