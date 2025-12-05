import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define parameters
referencing = "bip"
Hz = 10
length = 1000
n_permutation = "linear"
model_name = f"{referencing}_Hz_{Hz}_length_{length}_perm_{n_permutation}"

# Load the data (example DataFrame)
data = pd.read_csv(f"../../Decoding_results/{model_name}.csv")

# Get unique subjects
subjects = data["subject"].unique()

# Create bar plots for each subject
for subject in subjects:
    subject_data = data[data["subject"] == subject]
    x = np.arange(len(subject_data))  # Ensure x matches the length of subject_data
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (_, row) in zip(x, subject_data.iterrows()):  # Use zip to ensure proper indexing
        alpha = 1.0 if row["p"] <= 0.1 else 0.5
        ax.bar(i - width / 2, row["performance"], width, color="black", alpha=alpha, label='Performance' if i == 0 else "")
        ax.bar(i + width / 2, 1-row["p"], width, color="red", alpha=alpha, label='1-P-value' if i == 0 else "")
    ax.axhline(0.95, color="red", linestyle="--", label="0.05")

    ax.set_xlabel('Channels')
    ax.set_ylabel('Values')
    ax.set_title(f'Decoding Performance and P-values for subject {subject} model {model_name}', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(subject_data["channel"], fontsize=9)
    ax.tick_params(axis='x', labelsize=9)  # Set x-tick font size
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.savefig(f"../../Figures/{model_name}_performance_{subject}.png", dpi=300)
    #plt.close()
plt.show()