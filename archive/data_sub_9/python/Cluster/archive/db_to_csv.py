import sqlite3
import pandas as pd
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()  # Hide main window

db_path = filedialog.askopenfilename(title="Select .db file", filetypes=[("DB files", "*.db")])

# Connect to the database
conn = sqlite3.connect(db_path)

# Read the table into a pandas DataFrame
with sqlite3.connect(db_path) as conn:
    df = pd.read_sql("SELECT * FROM selected_data", conn)

# Save to CSV
df.to_csv(f'{db_path[:-3]}.csv', index=False)

# Close the connection
conn.close()
