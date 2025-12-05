%% For each electrode nii generate a list of overlapping fibers

% Prepare output storage
results = [];

flipped = "false";

subjects = unique(perf.subject);

for s = 1:length(subjects)
    subject = subjects(s);
    channels = unique(perf(perf.subject == subject, :).channel);

    for c = 1:length(channels)
        channel = channels(c);

        % Load the electrode NIfTI
        nii_path = sprintf('../../Electrode_nifti/Subject_%s_none_%s_flipped_%s.nii', num2str(subject), num2str(channel), flipped);
        nii = ea_load_nii(nii_path);

        % Convert to faces/vertices
        fv = ea_nii2fv(nii, 0.1);

        % Check which fibers intersect
        inside = inpolyhedron(fv.faces, fv.vertices, fibers(:, 1:3));

        % Get unique tract indices
        intersecting_idx = unique(fibers(inside, 4));

        % Append to results
        for idx = intersecting_idx'
            results = [results; {subject, channel, idx}];
        end
    end
end

% Convert results to table and save
results_table = cell2table(results, 'VariableNames', {'subject', 'channel', 'tract_idx'});
new_path = sprintf('tracts_overlap_%_%_%.csv', subject, channel, flipped);
writetable(results_table, new_path);

% nii = ea_load_nii('../../Electrode_nifti/Subject_1_none_13_flipped_false.nii');
% fv = ea_nii2fv(nii, 0.1);
% ea_mnifigure;
% true_idx = find(inpolyhedron(fv.faces, fv.vertices, fibers(1:100000, 1:3)));
% wjn_plot_fibers(fibers(true_idx, :));
% color = "#4f1025";
% patch('Faces',fv.faces,'Vertices',fv.vertices,'facecolor',color,'edgecolor',color, 'facealpha',0.2, 'edgealpha', 0);
