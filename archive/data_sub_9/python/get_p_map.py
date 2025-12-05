import numpy as np
from scipy.io import loadmat
import tkinter as tk
from tkinter import filedialog
import nibabel as nib

# Open file dialog
root = tk.Tk()
root.withdraw()
path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])

# Extract p map
corr_p = loadmat(path)
p = corr_p["p"]

# Save p map in nii format
template_path = f"../../Maps/Functional_connectivity/Subject_1_bipolar_1_func_seed_AvgR_Fz.nii"  # Assuming a path for NIfTI
nii = nib.load(template_path)
nii_new = nib.Nifti1Image(p, nii.affine, nii.header)
new_path = path[:-4] + "_p.nii"
nib.save(nii_new, new_path)

