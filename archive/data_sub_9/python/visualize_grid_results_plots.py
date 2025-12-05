import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import statsmodels.formula.api as smf

# Load data from SQLite
name = "results_func.db"
conn = sqlite3.connect(name)
df = pd.read_sql_query("SELECT * FROM selected_data", conn)
conn.close()

# Create a unique ID for each condition combination
df['combination'] = (
    df['subject_mode'].round(2).astype(str) + "_" +
    df['flip_mode'].astype(str) + "_" +
    df['overlap'].astype(str) + "_" +
    df['mask'].astype(str).str[:5]
)

# Drop nan values
df = df.dropna(subset=["similarity"])

df = df[df["subject_mode"].round(2) == 5.71]
df = df[df["overlap"].astype(str) == "None"]
df = df[df["flip_mode"] == "true"]

groups = [group for combo, group in df.groupby('combination') if len(group) >= 2]
n_groups = len(groups)

ncols = 3
nrows = (n_groups + ncols - 1) // ncols

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
axs = axs.flatten()

for i, group in enumerate(groups):
    combo = group['combination'].iloc[0]

    # Calculate Spearman correlation
    rho, pval = spearmanr(group['similarity'], group['accuracy'])

    # Scatter + regression line
    sns.regplot(data=group, x='similarity', y='accuracy', ax=axs[i], scatter_kws={'s':50})

    # Title as combination name
    axs[i].set_title(combo, fontsize=14, fontweight='bold')

    # Labels
    axs[i].set_xlabel("Similarity")
    axs[i].set_ylabel("Accuracy")

    # Remove top and right spines
    sns.despine(ax=axs[i], top=True, right=True)

    # Add correlation text as a legend box inside plot
    legend_text = f"Spearman rho = {rho:.2f}\np = {pval:.3f}"
    axs[i].legend([legend_text], loc='upper left', fontsize=10, frameon=True)

# Remove empty subplots
for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])

# Adjust spacing
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Save
plt.savefig(f"../../Figures/{name}_res.png", dpi=300)

plt.show()