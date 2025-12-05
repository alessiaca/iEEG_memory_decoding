% Save nifti files with fixed raius around channels

clear all;
close all;

addpath("C:\code\wjn_toolbox\")
addpath("C:\code\spm12\")
addpath(genpath("C:\code\leaddbs\"))

%figure = ea_mnifigure;
roi_paths = {};
% Loop over patient 
for sub=1:9
    path = sprintf('../data_epochs/metadata/Subject_0%s_electrode_locations_bipolar.csv', string(sub));
    coords = readtable(path); 
    
    n_electrodes = height(coords);
    cmap = [linspace(1, 0, n_electrodes)', zeros(n_electrodes, 1), linspace(0, 1, n_electrodes)'];
    indices = 1:n_electrodes;
    for el=1:length(indices)
        roi_path = sprintf('../data_epochs/metadata/nifti/Subject_0%s_bipolar_%s.nii', string(sub), string(el));
        mni = table2array(coords(indices(el), :));
        wjn_spherical_roi(roi_path, mni, 4);
        % Save the path
        roi_paths{end+1} = roi_path;
    end
end

%% Write roi paths to text file
fileID = fopen('../data_epochs/metadata/nifti/channel_nifit_bipolar.txt', 'w');

% Write each string to a new line
for i = 1:length(roi_paths)
    fprintf(fileID, '%s\n', roi_paths{i});
end
