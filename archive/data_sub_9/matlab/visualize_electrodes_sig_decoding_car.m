%% Visualize location of electrodes
clear all;
close all;

% Load the performance values
path = '../python/Cluster/WM_load_decoding_performance.csv';
perf = readtable(path); 

figure = ea_mnifigure;
count = 0;

all_pvals = [];
all_overlap = [];

% Loop over patient 
for sub=[1,2,3,5,6,7,8,9]
    %figure = ea_mnifigure;
    path = sprintf('../../Processed_data/metadata/Subject_0%s_electrode_locations.csv', string(sub));
    coords = readtable(path); 
    coords_matrix = table2array(coords);

    % Get the performance from one subject
    perf_sub = perf(perf.subject == sub, :);

    % Loop over electrodes
    n_electrodes = length(perf_sub.subject);
    % Define colors
    cmap = jet(256);  
    indices = round((perf_sub.accuracy - 0.2) * (size(cmap,1) - 1) / 0.3) + 1;
    %indices = round(1 + (length(cmap)-1) * overlap); 
    colors = ind2rgb(indices, cmap);
    for el=1:n_electrodes-1
        try
            % Compute the average position between the current and the next electrode
            mni = coords_matrix(el, :);
            if mni(1) > 0
                mni(1) = mni(1) * 1;
            end
            if perf_sub.pvalue(el) < 0.3
                alpha = 0.5;
            else
                alpha = 0.1;
            end
            wjn_plot_mni_roi(mni,0.7, colors(el, :, :), alpha);
        catch
        end
    end
end

%Add hippocampus
nii = ea_load_nii('../../Atlas/Automated Anatomical Labeling 3 (Rolls 2020)Hippocampus.nii');
fv = ea_nii2fv(nii, 0.1);
color = "#4f1025";
patch('Faces',fv.faces,'Vertices',fv.vertices,'facecolor',color,'edgecolor',color, 'facealpha',0.2, 'edgealpha', 0);

%Save plot
%saveas(gcf, sprintf('../figures/Subject_0%s_electrode_locations.png', string(sub)));

% Compute the overall correlation across all subjects
[R_total, P_total] = corr(all_pvals, all_overlap);

disp('--- Overall correlation across all subjects ---');
disp(['r = ', num2str(R_total)]);
disp(['p = ', num2str(P_total)]);