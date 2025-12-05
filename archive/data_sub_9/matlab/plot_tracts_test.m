%% Plot the tracts that pass through the electrode niftis

close all; 

% Load the tracts that intersect with the nii files
tract_overlap = readtable('..\python\tracts_overlap\1_33_false.csv');
% account for python (start at 0)
tract_overlap.tract_index = tract_overlap.tract_index + 1;
tract_use = unique(tract_overlap.tract_index);

%load('..\..\Atlas\HCP_1000_tracts.mat')
filtered_fibers = fibers(ismember(fibers(:,4), tract_use(end-100:end)), :);

% Plot the tracts which intersect
%ea_mnifigure;
wjn_plot_fibers(filtered_fibers);

% Plot the target
nii = ea_load_nii('../../Electrode_nifti/Subject_1_none_33_flipped_false.nii');
fv = ea_nii2fv(nii, 0.1);
color = "#4f1025";
patch('Faces',fv.faces,'Vertices',fv.vertices,'facecolor',color,'edgecolor',color, 'facealpha',0.2, 'edgealpha', 0);
