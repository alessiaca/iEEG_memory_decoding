%% Extract nifti files for hippocampus

atlas_path = '../other/Automated Anatomical Labeling 3 (Rolls 2020).nii';

atlas_nii = ea_load_nii(atlas_path);
nii_original = atlas_nii.img;

text = readtable(strcat(atlas_path(1:end-4),'.txt'));
target = 'Hippocampus';
idx = find(contains(text{:, 2}, target));
IDs = text{idx, 1};
    
% Save as one nifti file 
nii = atlas_nii;
new_img = ismember(nii_original, IDs);
nii.img = new_img;
nii.fname = strcat(atlas_path(1:end-4), target,'.nii');
ea_write_nii(nii)

% Save for left
ID_names = text{idx, 2};
IDs_left = IDs(endsWith(ID_names, "_L") | startsWith(ID_names, "Left"));
new_img = ismember(nii_original, IDs_left);
nii.img = new_img;
nii.fname = strcat(atlas_path(1:end-4), target,'_L.nii');
ea_write_nii(nii)

% Save for right
IDs_right = IDs(endsWith(ID_names, "_R") | startsWith(ID_names, "Right"));
new_img = ismember(nii_original, IDs_right);
nii.img = new_img;
nii.fname = strcat(atlas_path(1:end-4), target,'_R.nii');
ea_write_nii(nii)
