%% Visualize location of electrodes
clear all;
close all;

figure = ea_mnifigure;

hex = '#C34290';

% Loop over patient 
for sub=[1,2,3,5,6,7,8,9]
    path = sprintf('../../Processed_data/metadata/Subject_0%s_electrode_locations.csv', string(sub));
    coords = readtable(path); 
    coords_matrix = table2array(coords);
    % Loop over electrodes
    n_electrodes = length(coords_matrix);
    for el=1:n_electrodes-1
        % Compute the average position between the current and the next electrode
        mni = coords_matrix(el, :);
        if mod(el-1, 8) == 0
            wjn_plot_mni_roi(mni, 0.5, hex);
        else
            wjn_plot_mni_roi(mni, 0.5);
        end
        
    end
end

%Add hippocampus
nii = ea_load_nii('../../Atlas/Automated Anatomical Labeling 3 (Rolls 2020)Hippocampus.nii');
fv = ea_nii2fv(nii, 0.1);
color = "#4f1025";
%patch('Faces',fv.faces,'Vertices',fv.vertices,'facecolor',color,'edgecolor',color, 'facealpha',0.2, 'edgealpha', 0);

%Save plot
%saveas(gcf, sprintf('../figures/Subject_0%s_electrode_locations.png', string(sub)));