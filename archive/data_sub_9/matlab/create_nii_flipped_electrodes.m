% Create nii file from electrode positions
mode = "bipolar"; % or "normal"

% Initalize list used for functional connectivity map computation
roi_paths = {};

% Loop over patient 
for sub=1:9
    path = sprintf('../../Processed_data/metadata/Subject_0%s_electrode_locations.csv', string(sub));
    coords = readtable(path); 
    coords_matrix = table2array(coords);

    % Loop over electrodes
    n_electrodes = length(coords_matrix);
    for ele=1:n_electrodes-1
        if mode == "bipolar"
            % Compute the average position between the current and the next electrode
            mni = (coords_matrix(ele, :) + coords_matrix(ele+1, :)) / 2;
        else
            mni = coords_matrix(ele, :);
        end
        % If electrode number at edge (8th) don't safe
        if ~(mode == "bipolar" && mod(ele, 8) == 0)
            % Save the position as nii file with radius of 4 mm
            roi_path = sprintf('../../Electrode_nifti/Subject_%s_%s_%s.nii', string(sub), mode, string(ele));
            wjn_spherical_roi(roi_path, mni, 4);
    
            % Save the path
            roi_paths{end+1} = roi_path;
        end
    end
end

%% Write roi paths to text file
fileID = fopen(sprintf('../../Electrode_nifti/%s_list.txt', mode), 'w');

% Write each string to a new line
for i = 1:length(roi_paths)
    fprintf(fileID, '%s\n', roi_paths{i});
end