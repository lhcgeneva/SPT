%% This file is a wrapper for the Kilfoil particle tracking algorithm.
% To add a new folder for tracking see section 'Write parameters to file'. Images 
% have to be stored as 8-bit tifs in folders named fovn, where 'n' is the number 
% of the movie. After having created a log_file with parameters frameinterval, 
% featsize, masscut, maxdisp, min_track_length and memory for each movie, the 
% section 'Find particles' finds tracks for all movies in the current folder, 
% storing results in Bead_tracking and Feature_finding (One workspace per movie) 
% The last part of the code calculates off-rates using a method described in Robin 
% et al. (2014). For fitting off-rates, first call find_particles find_particles(logfile, 
% fovn, 0, 1); for feature finding and then change the parameters for time and 
% numbins in the last section of the code before running.
%% Write parameters to file
% When adding new files or initializing a new folder with movies, the following 
% stores some metadata that's important for the tracking in a file in the same 
% folder called 'log_file.txt'
% 
% frame_interval = [0.033; 0.033; 0.033; 0.033; 0.033]; % Frame interval 
% in s pixel_size = [0.12; 0.12; 0.12; 0.12; 0.12]; % pixel size in microns masscut 
% = [330; 350; 350; 350; 320]; featsize = [3; 3; 3; 3; 3]; maxdisp = [5; 5; 5; 
% 5; 5]; memory = [7; 7; 7; 7; 7]; min_track_length = [80; 80; 80; 80; 80]; log_file 
% = table(frame_interval, pixel_size, featsize, maxdisp, masscut, ... memory, 
% min_track_length); writetable(log_file);.
%% Find particles

logfile = readtable('log_file.txt');
for i = 1:height(logfile)
    find_particles(logfile, i, 0, 1)
end
%% Or: set parameters manually
% Use for testing individual movies, make sure all parameters are correct, then 
% run this section

fovn = 12;
featsize = nan(fovn, 1);
frame_interval = nan(fovn, 1);
masscut = nan(fovn, 1);
pixel_size = nan(fovn, 1);
maxdisp = nan(fovn, 1);
min_track_length = nan(fovn, 1);
memory = nan(fovn, 1);
featsize(end) = 3;
frame_interval(end) = 0.040;
masscut(end) = 200;
pixel_size(end) = 0.124;% SB: 0.1049;
% Tracking parameters
maxdisp(end) = 5;
min_track_length(end) = 80;
memory(end) = 7;
% Create table without writing to file (just for testing one movie)
log_file = table(frame_interval, pixel_size, featsize, maxdisp, masscut, ...
                 memory, min_track_length);
find_particles(log_file, fovn, 0, 1)
%% The following code can be run individually for each fovn (so for each movie)
% |By setting fovn=n the nth movie in this folder will be used for analysis. 
% The time step and pixel size are read from the metadata file 'log_file.txt'|

% logfile = readtable('log_file.txt');
logfile = log_file;
min_track_length = logfile.min_track_length(fovn);
frame_interval = logfile.frame_interval(fovn);

% Plot histogram of track lengths longer than min_track_length
figure(1); clf;
cd Bead_tracking/res_files;
load(['res_fov', num2str(fovn), '.mat']);
cd ../..
c = cell(1, max(res(:, 8)));
for i = 1:max(res(:, 8))
    c{i} = res(res(:, end) == i, :);
end
ls = cell2mat(cellfun(@(x) x(1), cellfun(@size, c, 'UniformOutput', false), ...
                                'UniformOutput', false));
edges = min_track_length-min_track_length:10:400-min_track_length;
figure(1)
h = histogram(ls, edges); shg; hold on;
title('Distribution of track lengths', 'FontSize', 16);
v = h.Values';
centers = (edges(1:end-1)+(edges(1)-edges(1))/2-80)';
v = v(centers>0);
centers = centers(centers>0);

% Put x_y positions for each track into cell array, add nans for missing points
pixelSize = logfile.pixel_size(fovn);
disp(['Pixel size is ', num2str(pixelSize), 'microns.']);
x_y_cell = cellfun(@(x) x(:, 1:2)*pixelSize, c, 'UniformOutput', false);
t_cell = cellfun(@(x) x(:, 7)-min(x(:, 7)), c, 'UniformOutput', false);
frame_index_cell = cellfun(@(x) x(:, 6), c, 'UniformOutput', false);
for j = 1:length(frame_index_cell)
    a = diff(frame_index_cell{j});
    i = 1;
    while i <= length(a)
        if ~isequal(a(i), 1)
            b = t_cell{j};
            d = x_y_cell{j};
            e = frame_index_cell{j};
            a = [a(1:i); ones(a(i)-1, 1); a(i+1:end)];
            t_cell{j} = [b(1:i); nan(a(i)-1, 1); b(i+1:end)];
            x_y_cell{j} = [d(1:i, :); nan(a(i)-1, 2); d(i+1:end, :)];
            frame_index_cell{j} = [e(1:i-1); (e(i):e(i+1))'; e(i+2:end)];
        end
        i = i + 1;
    end
end

% Get step distance between neighboring frames
distances = cellfun(@(x) sqrt(diff(x(:, 1)).^2 + diff(x(:, 2)).^2),...
                    x_y_cell, 'UniformOutput', false);

% Plot tracks, reversed y-axis (to fit image)
figure; hold on;
for i = 1:length(x_y_cell)
    plot(x_y_cell{i}(:, 1),x_y_cell{i}(:, 2));
end
set(gca,'Ydir','reverse');
axis square
title('Tracks superimposed on embryo', 'FontSize', 18);
xlabel('x ($\mu m$)', 'FontSize', 24, 'Interpreter', 'Latex');
ylabel('y ($\mu m$)', 'FontSize', 24, 'Interpreter', 'Latex');
%% Calculate msd from tracks
msd  = cell(1, length(c));
stepsize = cell(1, length(c));
for j = 1:length(x_y_cell)
    for tau = 1:length(x_y_cell{j})-1
        temp=nan(length(x_y_cell{j})-tau, 1);
        for i =  1:length(x_y_cell{j})-tau
            temp(i) = (x_y_cell{j}(i+tau, 1)-x_y_cell{j}(i, 1))^2+...
                          (x_y_cell{j}(i+tau,2)-x_y_cell{j}(i, 2))^2;
        end
        if tau == 1
            stepsize{j} = sqrt(temp);
        end
        msd{j}(tau) = nanmean(temp);
    end
    msd{j} = [0,msd{j}];
end
stepsize_rows = cellfun(@(x) x', stepsize, 'UniformOutput', false);
stepsize_mat = [stepsize_rows{:}];
steps_filter = stepsize_mat(stepsize_mat<0.1);
% Plot all msd over time
figure; hold on;
for i = 1:length(c)
    plot(msd{i}, 'b');
end
set(gca,'XScale','log');
set(gca,'YScale','log');
title('MSD over lag time', 'FontSize', 18);
xlabel('\tau [s]', 'FontSize', 24);
ylabel('\langle x^2 \rangle [um^2]', 'FontSize', 24);
%% 
% Fitting $\langle x^2 \rangle = 4\cdot D \cdot t^\alpha$, using a linear 
% fit in log-log space, in order to get best fitting performance/accuracy.
numFramesFit = 10;
D = zeros(1,length(msd));
a = zeros(1,length(msd));
for i = 1:length(msd)
    lx = log10((frame_interval):(frame_interval):(numFramesFit*frame_interval))';
    ly = log10(msd{i}(2:numFramesFit+1))';
    p = polyfit(lx, ly, 1);
    D(i) = 10^(p(2))/4;
    a(i) = p(1);
end
figure(6); clf; hold on;
scatter(a, D);
title('Anomalous Diffusion', 'FontSize', 18);
xlabel('\alpha', 'FontSize', 24);
ylabel('$D [\frac{\mu ^2}{s}]$','Interpreter','LaTex', 'FontSize', 24);
% axis([0 1.5 0 0.01]);
mean(D(a>0.9 & a<1.2))
%% Off Rate

cd Feature_finding/ load MT_1_Feat_Size_5.mat;
timestep = 0.5; %(time step in seconds)
numBins = 150; 
data = histcounts(MT(:, 7), numBins); 
fitData = data/max(data); 
fitData = fitData(1:numBins); 
fitTimes = 0:timestep:((length(fitData)-1)*timestep); 
figure; hold on; plot(fitTimes, fitData); 
x = fit_offRate(fitTimes, fitData); 
cd ..
%% Create movie with tracked particles in red
% Read images

numFrames = 500;
cd(['fov', num2str(fovn)])
im = imread(['fov', num2str(fovn), '_0001.tif']);
im = zeros([numFrames, size(im)]);
for i = 1:numFrames
    im(i, :, :) = imread(['fov', num2str(fovn), '_', ...
                            num2str(i,'%04d'), '.tif']);
end
cd ..
%% 
% Display movie

cd('Bead_tracking/res_files/');
load(['res_fov', num2str(fovn), '.mat']);
cd ../..
v = VideoWriter('Tracking_50p.avi');
open(v);
figure('Visible','Off')
for i = 1:numFrames
    imshow(squeeze(im( i, :, :)) ,[]);
    hold on;
    plot(res(res(:, 6) == i, 1),res(res(:, 6) == i, 2), 'ro',...
         'MarkerSize', 5);
    hold off;
    frame = getframe;
    writeVideo(v,frame);
drawnow
end
close(v)