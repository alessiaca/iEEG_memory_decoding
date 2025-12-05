%% Visualize location of electrodes
clear all;
close all;

% Load the performance values
path = '../../Decoding_results/bip_Hz_5_length_1000_runs_500.csv';
perf = readtable(path); 

figure = ea_mnifigure;
count = 0;

% Loop over patient 
for sub=1:9
    %figure = ea_mnifigure;
    path = sprintf('../../Processed_data/metadata/Subject_0%s_electrode_locations_bip.csv', string(sub));
    coords = readtable(path); 
    coords_matrix = table2array(coords);

    % Get the performance from one subject
    perf_sub = perf(perf.subject == sub, :);

    % Loop over electrodes
    n_electrodes = length(coords_matrix);
    % Define colors
    cmap = plasma(256);  
    indices = round((perf_sub.accuracy - 0.5) * (size(cmap,1) - 1) / 0.3) + 1;
    colors = ind2rgb(indices, cmap);
    for el=1:n_electrodes
        mni = coords_matrix(el, 2:end);
        % Flip from right to left hemisphere
        if mni(1) > 0
            mni(1) = mni(1) * -1;
        end
        % Plot only channels of interest
        if perf_sub.p_value(el) < 0.05/height(perf_sub) && perf_sub.accuracy(el) > 0.5
            alpha = 1;
        else
            alpha = 0.1;
        end
        % Compute the average position between the current and the next electrode
        wjn_plot_mni_roi(mni,0.4, colors(el, :, :), alpha);
    end
end

%Add hippocampus
nii = ea_load_nii('../../Atlas/Automated Anatomical Labeling 3 (Rolls 2020)Hippocampus.nii');
fv = ea_nii2fv(nii, 0.1);
color = "#4f1025";
patch('Faces',fv.faces,'Vertices',fv.vertices,'facecolor',color,'edgecolor',color, 'facealpha',0.2, 'edgealpha', 0);
