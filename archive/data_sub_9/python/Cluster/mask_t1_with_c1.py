import nibabel as nib
import numpy as np

# Load T1 and GM images
t1_img = nib.load("../../../Atlas/t1.nii")
gm_img = nib.load("../../../Atlas/rc1mask.nii")

t1_data = t1_img.get_fdata()
gm_data = gm_img.get_fdata()

# Apply threshold to GM mask (e.g., 0.3)
gm_mask = gm_data > 0.3

# Mask T1 image
masked_t1_data = t1_data * gm_mask

# Save the masked image
masked_img = nib.Nifti1Image(masked_t1_data, t1_img.affine, t1_img.header)
nib.save(masked_img, "T1_masked_by_GM.nii")
