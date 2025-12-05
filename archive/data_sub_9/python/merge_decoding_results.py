import sqlite3
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import glob

def merge_sqlite_dbs():
    # Open folder picker
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select folder with .db files")
    if not folder_selected:
        print("No folder selected.")
        return

    # Get all .db files
    db_files = glob.glob(os.path.join(folder_selected, "*.db"))
    if not db_files:
        print("No .db files found in the selected folder.")
        return

    # Collect all data
    all_data = []
    for db_file in db_files:
        try:
            with sqlite3.connect(db_file) as conn:
                df = pd.read_sql("SELECT * FROM selected_data", conn)
                df["source_db"] = os.path.basename(db_file)  # Optional: keep track of origin
                all_data.append(df)
        except Exception as e:
            print(f"Failed to read {db_file}: {e}")

    if not all_data:
        print("No data loaded.")
        return

    # Combine and save
    merged_df = pd.concat(all_data, ignore_index=True)

    # Output merged database
    merged_db_path = os.path.join(folder_selected, "merged_results.db")
    with sqlite3.connect(merged_db_path) as conn:
        merged_df.to_sql("selected_data", conn, index=False, if_exists="replace")

    print(f"Merged {len(db_files)} databases into {merged_db_path}.")

if __name__ == "__main__":
    merge_sqlite_dbs()
