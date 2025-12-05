%% Visualize location of electrodes
clear all;
close all;

count = 0;
paths_sig = {};
% Loop over patient 
for sub=1:9 %[3,5,6,7,8,9]

    % Load the performance values
    path = sprintf('perf_%s.csv', string(sub));
    perf = readtable(path); 
    for el=1:height(perf)
        if perf(el, 3).p < 0.05
            % Save the path to the functional connectivity map for a
            % significant channel 
            paths_sig{end+1} = sprintf('../data_epochs/metadata/nifti/fmaps/Subject_0%s_bipolar_%s_conn-PPMI74P15CxControls_desc-AvgR_funcmap.nii', string(sub), string(el));
        end
    end
end

%% Average the functional connectivity maps
num_files = length(paths_sig);
expression = sprintf('(%s) / %d', ...
    strjoin(arrayfun(@(i) sprintf('i%d', i), 1:num_files, 'UniformOutput', false), ' + '), num_files);
spm_imcalc(char(paths_sig'), '../data_epochs/metadata/nifti/fmaps/average_sig_conn-PPMI74P15CxControls_desc-AvgR_funcmap.nii', expression);
