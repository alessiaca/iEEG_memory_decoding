% Add and resample ROI nii

% Select NIfTI files
path1 = 'C:/Users/ICN/OneDrive - Charité - Universitätsmedizin Berlin/PROJECT Memory Decoding/Atlas/nucleus_accumbens_lh.nii';
path2 = 'C:/Users/ICN/OneDrive - Charité - Universitätsmedizin Berlin/PROJECT Memory Decoding/Atlas/nucleus_accumbens_rh.nii';

% Baseline file 
base_file = '../../Maps/Functional_connectivity/Subject_1_bipolar_2_func_seed_AvgR_Fz.nii';

% Create new resampled file
new_path =  'C:/Users/ICN/OneDrive - Charité - Universitätsmedizin Berlin/PROJECT Memory Decoding/Atlas/nucleus_accumbens.nii';
expression = sprintf('i2 + i3');
flags =struct();
flags.mask = -1;
spm_imcalc(char(base_file, path1, path2), new_path, expression, flags);
