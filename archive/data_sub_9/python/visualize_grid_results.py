import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import statsmodels.formula.api as smf

# Load data from SQLite
conn = sqlite3.connect("results_func.db")
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
df = df[df["overlap"].astype(str) == "0.0"]
df = df[df["flip_mode"] == "true"]


# Compute the correlation across all subjects and channels
corr = []
for combo, group in df.groupby('combination'):
    if len(group) >= 2:  # Spearman requires at least 2 values
        rho, pval = spearmanr(group['similarity'], group['accuracy'])
        corr.append({'combination': combo, 'rho': rho, 'pval': pval})

corr_df = pd.DataFrame(corr).sort_values('rho', ascending=False)

# Plot Spearman correlations with transparency by p-value
fig, axes = plt.subplots(2, 1,figsize=(18, 12))
bars = sns.barplot(x='combination', y='rho', data=corr_df, palette='plasma', ax=axes[0])
for bar, pval in zip(bars.patches, corr_df['pval']):
    bar.set_alpha(1.0 if pval < 0.05 else 0.2)
#axes[0].set_xticklabels([])
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90, fontsize=7)
axes[0].set_ylabel("Spearman Correlation")
axes[0].set_xlabel("")
sns.despine(ax=axes[0])

# Compute a LME accounting for subject individual intercepts
lme_results = []
for combo, group in df.groupby('combination'):
    model = smf.mixedlm("accuracy ~ similarity", data=group, groups=group["subject"])
    result = model.fit(method="nm")
    coef = result.params["similarity"]
    pval = result.pvalues["similarity"]
    lme_results.append({'combination': combo, 'coef': coef, 'pval': pval})

lme_df = pd.DataFrame(lme_results).sort_values('coef', ascending=False)

# Plot LME coefficients with transparency by p-value
bars = sns.barplot(x='combination', y='coef', data=lme_df, palette='cividis', ax=axes[1])
for bar, pval in zip(bars.patches, lme_df['pval']):
    bar.set_alpha(1.0 if pval < 0.05 else 0.2)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90, fontsize=7)
axes[1].set_ylabel("LME Coefficient")
sns.despine(ax=axes[1])
plt.subplots_adjust(hspace=1, wspace=1, bottom=0.3)
plt.show()