%% Visualize location of electrodes
clear all;
close all;

figure = ea_mnifigure;
% Loop over patient 
for sub=1:9
    path = sprintf('../data_epochs/metadata/Subject_0%s_electrode_locations_bipolar.csv', string(sub));
    coords = readtable(path); 
    %figure = ea_mnifigure;

    % Load the performance values
    path = sprintf('perf_%s.csv', string(sub));
    perf = readtable(path); 

    % Select the colors for the performance values
    cmap = cool(256);  

    % Normalize values to fit between 1 and 256 (colormap indices)
    normalized_perf = round(rescale(1-perf.p, 1, 256));

    % Get the corresponding colors from the colormap
    colors = cmap(normalized_perf, :);

    n_electrodes = height(coords);
    cmap = [linspace(1, 0, n_electrodes)', zeros(n_electrodes, 1), linspace(0, 1, n_electrodes)'];
    indices = 1:n_electrodes;
    for el=1:length(indices)
        mni = table2array(coords(indices(el), :));
        color = "#808080";
        alpha = 0.1;
        if perf(el, 3).p < 0.05
            color = "#FF1493";
            alpha = 1; 
        end
        wjn_plot_mni_roi(mni,1, colors(el, :), alpha);
    end
    
    % Add hippocampus
    nii = ea_load_nii('../other/Automated Anatomical Labeling 3 (Rolls 2020)Hippocampus.nii');
    fv = ea_nii2fv(nii, 0.1);
    color = "#4f1025";
    patch('Faces',fv.faces,'Vertices',fv.vertices,'facecolor',color,'edgecolor',color, 'facealpha',0.2, 'edgealpha', 0);

    % Save plot
    saveas(gcf, sprintf('../figures/Subject_0%s_electrode_locations.png', string(sub)));

end

