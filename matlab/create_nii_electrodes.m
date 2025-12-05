% Create nii file from electrode positions
mode = "bipolar";
flip_right = true;

% Initalize list used for functional connectivity map computation
roi_paths = {};

% Loop over patient 
for sub=1:15
    if mode == "bipolar"
        path = sprintf('../../Data_processed/metadata/Subject_%s_bipolar_electrode_locations.csv', string(sub));
    else
        path = sprintf('../../Data_processed/metadata/Subject_%s_electrode_locations.csv', string(sub));
    end
    coords = readtable(path); 

    % Loop over electrodes
    n_electrodes = height(coords);
    for ele=1:n_electrodes
        mni = table2array(coords(ele, 2:4));
        
        % Flip to other hemisphere
        if flip_right == true && mni(1) > 0
            mni(1) = mni(1) * -1;
            flipped = true;
        else
            flipped = false;
        end

        % Save the position as nii file with radius of 4 mm
        roi_path = sprintf('../../Electrode_nifti/Subject_%s_%s_%s_flipped_%s.nii', string(sub), mode, coords{ele,1}{1}, string(flipped));
        wjn_spherical_roi(roi_path, mni, 3);
    
        % Save the path
        roi_paths{end+1} = roi_path;
    end
end

%% Write roi paths to text file
fileID = fopen(sprintf('../../Electrode_nifti/%s_list_flipped_%s.txt', mode, string(flip_right)), 'w');

% Write each string to a new line
for i = 1:length(roi_paths)
    fprintf(fileID, '%s\n', roi_paths{i});
end