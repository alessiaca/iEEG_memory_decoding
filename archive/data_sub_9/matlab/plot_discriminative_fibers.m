%% Plot discriminatiev fibers
close all; 

%% Load the fiber tract
load('..\..\Atlas\HCP_1000_tracts.mat')

%%
% Load the statistic for every fiber
fiber_stats = readtable('..\..\Fiber_filtering\t_p_fibers_true_0.25_5.714285714285714.csv');
fiber_stats.tract_index = fiber_stats.tract_index + 1;
tract_use = fiber_stats((fiber_stats.p_value < 0.05) & (fiber_stats.ratio > 0.1) & (fiber_stats.t_statistic < 0), :).tract_index;

filtered_fibers = fibers(ismember(fibers(:,4), tract_use), :);

% Plot the tracts which intersect
ea_mnifigure;
wjn_plot_fibers(filtered_fibers);
