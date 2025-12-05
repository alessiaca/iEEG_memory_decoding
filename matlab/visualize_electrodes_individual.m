%% Visualize location of electrodes
close all; 
clear all;
% Loop over patient 
for sub=1
    path = sprintf('../../Processed_data/metadata/Subject_%s_electrode_locations.csv', string(sub));
    coords = readtable(path); 
    figure = ea_mnifigure;
    
    n_electrodes = height(coords);
    cmap = [linspace(1, 0, n_electrodes)', zeros(n_electrodes, 1), linspace(0, 1, n_electrodes)'];
    indices = 2:n_electrodes;
    for el=1:length(indices)
        mni = table2array(coords(indices(el), 2:4));
        wjn_plot_mni_roi(mni,0.75, "#C34290");
    end  
    % Add hippocampus
    nii = ea_load_nii('../../Atlas/Automated Anatomical Labeling 3 (Rolls 2020)Hippocampus.nii');
    fv = ea_nii2fv(nii, 0.1);
    color = "#4f1025";
    patch('Faces',fv.faces,'Vertices',fv.vertices,'facecolor',color,'edgecolor',color, 'facealpha',0.2, 'edgealpha', 0);

    % Save plot
    %saveas(gcf, sprintf('../../Figures/Subject_%s_electrode_locations.png', string(sub)));

    %close all;
end

