clear
lead path % add Lead-DBS to the path

resultfig = ea_mnifigure; % Create empty 3D viewer figure

%%
% Get the paths to thr ROIs
path = '../../Decoding_results/none_Hz_5_length_1000_runs_200.csv';
perf = readtable(path); 
flipped = false;
mode = "none";

ROI_paths = {};
accuracy = [];

% Loop over patient 
for sub=[2,3,5,6,7,8,9]
    % Get the performance from one subject
    perf_sub = perf(perf.subject == sub, :);
    n_electrodes = length(perf_sub.subject);
    for el=1:n_electrodes
        accuracy_sub = perf_sub(el, :).accuracy;
        if accuracy_sub > 0.5
            ROI_path = sprintf('../../Electrode_nifti/Subject_%s_%s_%s_flipped_%s.nii', string(sub), mode, string(el), string(flipped));
            ROI_paths{end+1} = ROI_path;
            accuracy(end+1) = accuracy_sub;
        end
    end
end
%%
resultfig = ea_mnifigure;
M.pseudoM = 1; % Declare this is a pseudo-M struct, i.e. not a real lead group file
M.ROI.list=ROI_paths(1:100)';

M.ROI.group=ones(length(M.ROI.list),1);

M.clinical.labels={'Improvement','Covariate_1','Covariate_2'}; % how will variables be called
M.clinical.vars{1}= accuracy(1:100)';

M.guid='My_Analysis'; % give your analysis a name

save('Analysis_Input_Data.mat','M'); % store data of analysis to file

% Open up the Sweetspot Explorer
ea_sweetspotexplorer(fullfile(pwd,'Analysis_Input_Data.mat'),resultfig);

% Open up the Network Mapping Explorer
%ea_networkmappingexplorer(fullfile(pwd,'Analysis_Input_Data.mat'),resultfig);

% Open up the Fiber Filtering Explorer
%ea_discfiberexplorer(fullfile(pwd,'Analysis_Input_Data.mat'),resultfig);





