import tkinter as tk
from tkinter import filedialog
import sqlite3
import pandas as pd

# Open file chooser dialog
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window

file = filedialog.askopenfilename(
    title="Select SQLite Database File",
    filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")]
)

if file:
    # Connect to SQLite and read data
    conn = sqlite3.connect(file)
    df = pd.read_sql("SELECT * FROM selected_data", conn)

    # Save as CSV (same name, different extension)
    csv_file = file[:-3] + ".csv"
    df.to_csv(csv_file, index=False)

    conn.close()
    print(f"Data saved to {csv_file}")
else:
    print("No file selected.")
