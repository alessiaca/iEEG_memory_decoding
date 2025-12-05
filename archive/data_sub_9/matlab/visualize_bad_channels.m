%% Visualize location of electrodes
clear all;
close all;

figure = ea_mnifigure;

% Read the bad channels
bad_channels = load('../../Processed_data/metadata/bad_channels_mask.mat');

% Loop over patient 
for sub=1:9
    % Load electrode locations
    %figure = ea_mnifigure;
    path = sprintf('../../Processed_data/metadata/Subject_0%s_electrode_locations.csv', string(sub));
    coords = readtable(path); 
    coords_matrix = table2array(coords);

    % Load overlap with grey matter mask
    path = sprintf('../../Processed_data/metadata/Subject_%s_overlap.csv', string(sub));
    overlap = readtable(path).percent_overlap; 

    % Get the bad channels for the subject
    sub_field = sprintf('subject_%d', sub);
    bad_channels_sub = bad_channels.bad_channels.(sub_field);

    % Define color based on overlap 
    cmap = plasma(256);  
    idx = round(1 + (length(cmap)-1) * overlap); 
    colors = cmap(idx, :);

    % Loop over electrodes
    n_electrodes = length(coords_matrix);
    for el=1:n_electrodes

        % If electrode number not at edge (8th) 
        try
            % Compute the average position between the current and the next electrode
            mni = coords_matrix(el, :);
            if overlap(el) < 0.1 %ismember(el, bad_channels_sub) | true
                alpha = 1;
            else
                alpha = 0.1;
            end
            wjn_plot_mni_roi(mni, 0.4, colors(el, :, :), alpha);
        catch
        end
    end
    
    % Add hippocampus
    nii = ea_load_nii('../../Atlas/Automated Anatomical Labeling 3 (Rolls 2020)Hippocampus.nii');
    fv = ea_nii2fv(nii, 0.1);
    color = "#4f1025";
    patch('Faces',fv.faces,'Vertices',fv.vertices,'facecolor',color,'edgecolor',color, 'facealpha',0.2, 'edgealpha', 0);

    % Save plot
    %saveas(gcf, sprintf('../figures/Subject_0%s_electrode_locations.png', string(sub)));

end

