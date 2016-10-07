function [distances, result_cell, t_cell, x_y_cell] =...
            get_x_y(fovn, logtable, PLOTTING, h_thist, h_tracks, h_steps)
%GET_X_Y Extracts x-y positions from each track, calculates step distance
%        GET_X_Y(fovn, logtable, POTTING) Plots all tracks color coded 
%        and histogram of track lengths.
%        GET_X_Y(fovn, logtable, PLOTTING, h_thist, h_tracks) also takes
%        handles for where to put histogram of track lengths, tracks amd
%        histogram of stepsizes

%Check for right number of input arguments
if ~(nargin == 3 || nargin == 6);
    disp('Wrong number of input arguments, type help get_x_y.');
    distances = {};
    result_cell = {};
    t_cell = {};
    x_y_cell = {};
    return;
end
% Set parameters, load tracking data
min_track_length = logtable.min_track_length(fovn);
cd Bead_tracking/res_files;
load(['res_fov', num2str(fovn), '.mat']);
cd ../..
% Convert tracking matrix into cell array, each track is one cell
result_cell = cell(1, max(res(:, 8)));
for i = 1:max(res(:, 8))
    result_cell{i} = res(res(:, end) == i, :);
end
% Get length of each track and calculate edges for histogram plot
ls = cell2mat(cellfun(@(x) x(1), cellfun(@size, result_cell,...
              'UniformOutput', false), 'UniformOutput', false));
% Put x_y positions tracks into cell array, nans for missing points
pixelSize = logtable.pixel_size(fovn);
x_y_cell = cellfun(@(x) x(:, 1:2)*pixelSize, result_cell,...
                   'UniformOutput', false);
t_cell = cellfun(@(x) x(:, 7)-min(x(:, 7)), result_cell,...
                 'UniformOutput', false);
% Get step distance between neighboring frames
distances = cellfun(@(x) (sqrt(diff(x(:, 1)).^2+diff(x(:, 2)).^2))',...
                          x_y_cell, 'UniformOutput', false);                      
if PLOTTING        
    % Histogram of track lengths
    if nargin == 3
        figure();
    else
        axes(h_thist);
    end
    histogram(ls, 'BinWidth', 10, 'EdgeColor', 'none'); shg; hold on;
    box off;
    title('Track length distribution', 'FontSize', 14)
    xlabel('Track length', 'FontSize', 16);
    ylabel('# tracks', 'FontSize', 16); 
    
    % Plot tracks, reversed y-axis (to fit image)
    if nargin == 3
        figure(); hold on;
    else
        axes(h_tracks); hold on;
    end
    for i = 1:length(x_y_cell)
        plot(x_y_cell{i}(:, 1),x_y_cell{i}(:, 2));
    end
    set(gca,'Ydir','reverse');
    axis square
    title(['Tracks with more than ', num2str(min_track_length),...
           'steps'], 'FontSize', 14)
    ylabel('y [\mum]', 'FontSize', 16);
    xlabel('x [\mum]', 'FontSize', 16); 
    
    % Histogram of step sizes
    if nargin == 3
        figure();
    else
        axes(h_steps);
    end
    histogram([distances{:}], 'EdgeColor', 'none');
    box off;
    title('Step size distribution', 'FontSize', 14)
    xlabel('Step size [\mum]', 'FontSize', 16);
    ylabel('# Steps', 'FontSize', 16);
end