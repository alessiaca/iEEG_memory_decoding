%% Visualize location of electrodes
clear all;
close all;

% Prepare plotting
figure = ea_mnifigure;
cmap = jet(256);  

% Load the decoding accuracies
path = sprintf('../Python/Decoding/decoding_accuracies.csv');
accuracies = readtable(path); 

% Normalize values to fit between 1 and 256 (colormap indices)
normalized_perf = round(rescale(1-accuracies.accuracy, 1, 256));

% Loop over patients 
count = 1;
for sub=1:15
    path = sprintf('../../Data_processed/Metadata/Subject_%s_bipolar_electrode_locations.csv', string(sub));
    coords = readtable(path); 
    
    % Loop over channels
    for el=1:height(coords)
        mni = table2array(coords(el, 2:4));
        if mni(1) > 0
            mni(1) = mni(1) * -1;
        end
        try
            wjn_plot_mni_roi(mni,0.5, cmap(normalized_perf(count),:), 0.5);
        catch
            disp("Error")
        end
        count = count + 1;
    end  
end

% Add hippocampus
nii = ea_load_nii('../../Atlas/Automated Anatomical Labeling 3 (Rolls 2020)Hippocampus.nii');
fv = ea_nii2fv(nii, 0.1);
color = "#4f1025";
patch('Faces',fv.faces,'Vertices',fv.vertices,'facecolor',color,'edgecolor',color, 'facealpha',0.2, 'edgealpha', 0);

% Save plot
%saveas(gcf, sprintf('../figures/Subject_0%s_electrode_locations.png', string(sub)));

