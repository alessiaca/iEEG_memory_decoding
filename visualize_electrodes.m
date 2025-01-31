%% Visualize location of electrodes

% Loop over patient 
for sub=1:9
    path = sprintf('../data_epochs/Subject_0%s_electrode_locations.csv', string(sub));
    coords = readtable(path); 
    figure = ea_mnifigure;
    
    n_electrodes = height(coords);
    for el=1:n_electrodes
        mni = table2array(coords(el, :));
        wjn_plot_mni_roi(mni,1,[0.5, 0, 0.5])
    end
    
    % Add hippocampus
    nii = ea_load_nii('../other/Automated Anatomical Labeling 3 (Rolls 2020)Hippocampus.nii');
    fv = ea_nii2fv(nii, 0.1);
    color = "#4f1025";
    patch('Faces',fv.faces,'Vertices',fv.vertices,'facecolor',color,'edgecolor',color, 'facealpha',0.2, 'edgealpha', 0);

    % Save plot
    saveas(gcf, sprintf('../figures/Subject_0%s_electrode_locations.png', string(sub)));

    close all;
end

