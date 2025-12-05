%% Visualize location of electrodes
close all; 
clear all;
figure = ea_mnifigure;
colors = ["#FF1493", "#1E90FF", "#32CD32", "#FFD700", "#8A2BE2", "#FF4500", "#00CED1", "#DC143C", "#7B68EE"];

% Loop over patient 
for sub=1:9
    path = sprintf('../../Processed_data/metadata/Subject_0%s_electrode_locations.csv', string(sub));
    coords = readtable(path); 
    
    n_electrodes = height(coords);
    cmap = [linspace(1, 0, n_electrodes)', zeros(n_electrodes, 1), linspace(0, 1, n_electrodes)'];
    indices = 1:n_electrodes;
    for el=1:length(indices)
        mni = table2array(coords(indices(el), :));
        wjn_plot_mni_roi(mni,1, colors(sub), 0.5);
    end  
end
% Add hippocampus
nii = ea_load_nii('../../Atlas/Automated Anatomical Labeling 3 (Rolls 2020)Hippocampus.nii');
fv = ea_nii2fv(nii, 0.1);
color = "#4f1025";
patch('Faces',fv.faces,'Vertices',fv.vertices,'facecolor',color,'edgecolor',color, 'facealpha',0.2, 'edgealpha', 0);

