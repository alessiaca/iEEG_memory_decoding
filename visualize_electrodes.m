%% Visualize location of electrodes


% Loop over patient 
for sub=1:9
    path = sprintf('../data_epochs/metadata/Subject_0%s_electrode_locations.csv', string(sub));
    coords = readtable(path); 
    figure = ea_mnifigure;
    
    n_electrodes = height(coords);
    cmap = [linspace(1, 0, n_electrodes)', zeros(n_electrodes, 1), linspace(0, 1, n_electrodes)'];
    indices = 1:8:n_electrodes;
    for el=1:length(indices)
        mni = table2array(coords(indices(el), :));
        wjn_plot_mni_roi(mni,1, "#4f1025");
    end
    indices = 2:8:n_electrodes;
    for el=1:length(indices)
        mni = table2array(coords(indices(el), :));
        wjn_plot_mni_roi(mni,1, '#00FF00');
    end
    
    % Add hippocampus
    nii = ea_load_nii('../other/Automated Anatomical Labeling 3 (Rolls 2020)Hippocampus.nii');
    fv = ea_nii2fv(nii, 0.1);
    color = "#4f1025";
    patch('Faces',fv.faces,'Vertices',fv.vertices,'facecolor',color,'edgecolor',color, 'facealpha',0.2, 'edgealpha', 0);

    % Save plot
    saveas(gcf, sprintf('../figures/Subject_0%s_electrode_locations.png', string(sub)));

    %close all;
end

