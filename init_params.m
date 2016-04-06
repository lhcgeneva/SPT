clear all
pixelSize = 0.155;
timestep = 0.030;
numFramesFit = 10;
basepath=[pwd, '/'];
fovn = 5;
featsize = 3;
barint = 1;
barrg = 50;
barcc = 1;
IdivRg = 0;
numFrames = 99;
Imin = 0;
masscut = 550;
field = 2;
frame = 99;
[M2, MT] = mpretrack_init( basepath, featsize, barint,...
    barrg, barcc, IdivRg, fovn, frame, Imin, masscut, field);
drawnow;
%% 
mpretrack(basepath, fovn, featsize, barint, barrg, ...
            barcc, IdivRg, numFrames, Imin, masscut, field );
%%
maxdisp = 5;
goodenough = 80;
memory = 7;
for fovn = 1:6  
    fancytrack(basepath, fovn, featsize, maxdisp, goodenough, memory );
end
%%
    figure(1);
    clf;
for fovn = 1:6
    cd Bead_tracking/res_files;
    load(['res_fov', num2str(fovn), '.mat']);
    cd ../..
    c = cell(1, max(res(:, 8)));
    for i = 1:max(res(:, 8))
    c{i} = res(res(:, end) == i, :);
    end
    ls = cell2mat(cellfun(@(x) x(1), cellfun(@size, c, 'UniformOutput', false), 'UniformOutput', false));
    edges = goodenough-goodenough:10:400-goodenough;
    figure(1)
    h = histogram(ls, edges); shg; hold on;
%     %Fit power law
%     f=('a*(x-b)^c');
%     options = fitoptions(f);
%     options.StartPoint = [500, 0, -1];
%     centers = (edges(1:end-1)+(edges(1)-edges(1))/2-80)';
%     v = h.Values';
%     v = v(centers>0);
%     centers = centers(centers>0);
%     fi = fit(centers, v, f, options);
%     fi.c
    %Fit exponential
%     f=('a*exp(-(x-b)/c)');
%     options = fitoptions(f);
%     options.StartPoint = [70, 0, 10];
%     centers = (edges(1:end-1)+(edges(1)-edges(1))/2-80)';
%     v = h.Values';
%     v = v(centers>0);
%     centers = centers(centers>0);
%     fi = fit(centers, v, f, options);
%     fi.c
%     close
%     % Plot
%     figure(2); 
%     subplot(3, 2, fovn); hold on;
% %     h = histogram(ls, edges-fi.c); shg; hold on;
%     plot(centers, v, '.', 'MarkerSize', 15)
%     plot(fi);
%     l = legend;
%     set(l,'visible','off')
%     text(150, 50, num2str(fi.c), 'FontSize', 16);
end
%% Read images
cd([basepath, 'fov', num2str(fovn)])
im = imread(['fov', num2str(fovn), '_0001.tif']);
im = zeros([numFrames, size(im)]);
for i = 0:1000%numFrames
    im(i+1, :, :) = imread(['fov', num2str(fovn), '_', num2str(i,'%04d'), '.tif']);
end
cd ..
%% Display movie
v = VideoWriter('Tracking_50p.avi');
open(v);
figure('Visible','Off')
for i = 1:1000%numFrames
    imshow(squeeze(im( i, :, :)) ,[]);
    hold on;
    plot(res(res(:, 6) == i, 1),res(res(:, 6) == i, 2), 'ro', 'MarkerSize', 15);
    hold off;
    frame = getframe;
    writeVideo(v,frame);
drawnow
end
close(v)
%% Calculate simulated brownian motion
Diff_coeffs = normrnd(0.15, 0.05, 177);
x_y_cell = cell(1, 177);
for i = 1:177
    x_y_cell{i}(1, 1:2) = 0;
    for j = 2:80
        x_y_cell{i}(j, 1:2) = x_y_cell{i}(j-1, 1:2) + ...
                                randn(1, 2)*sqrt(2*Diff_coeffs(i)*timestep);
    end
end
t_cell = repmat({linspace(0, timestep*80, 80)'}, 1, 177);
%% Get x_y positions from experiments
x_y_cell = cellfun(@(x) x(:, 1:2)*pixelSize, c, 'UniformOutput', false);
t_cell = cellfun(@(x) x(:, 7)-min(x(:, 7)), c, 'UniformOutput', false);
%% Plot tracks, reversed y-axis (to fit image)
figure; hold on;
for i = 1:length(x_y_cell)
plot(x_y_cell{i}(:, 1),x_y_cell{i}(:, 2));
end
set(gca,'Ydir','reverse')
%% Calculate MSD via gaussians
d = cell(1, length(c));
options = statset('Display','final', 'MaxIter', 1000);
D = zeros(1, length(c));
for i = 1:length(c)
    d{i} = (x_y_cell{i}(:, 1) - x_y_cell{i}(1, 1)).^2 + (x_y_cell{i}(:, 2) - x_y_cell{i}(1, 2)).^2;
    obj = gmdistribution.fit(diff(d{i}),1,'Options',options);
    D(i) = obj.mu;
end
%% Plot all msd over time
figure; hold on;
for i = 1:length(c)
    plot(t_cell{i}, d{i}, 'b');
end
set(gca,'XScale','log');
set(gca,'YScale','log');
%% Write msd to file (for python)
to_write = cell2mat(cellfun(@(x) x(1:num_Frames), d, 'UniformOutput', false));
dlmwrite('msd_matlab.csv', to_write);
%% load python data
python_msd = dlmread('msd_python.csv');
sz = size(python_msd);
d_python = mat2cell(python_msd, sz(1), ones(1, sz(2)));
t_cell_python = cellfun(@(x) linspace(timestep, length(x)*timestep, length(x))', ...
                    d_python, 'UniformOutput', false);
%% Fit MSD = 4Dt^a
clear a;
clear D;
clear d_fit;
clear t_fit;
d_fit = d;
t_fit = t_cell;
fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',[0 0],...
               'Upper',[Inf Inf],...
               'StartPoint',[0.15 1]);
ft = fittype('4*D*x^a', 'options', fo);
D = zeros(1,length(d_fit));
a = zeros(1,length(d_fit));
for i = 1:length(d_fit)
    f = fit(t_fit{i}(1:numFramesFit),d_fit{i}(1:numFramesFit), ft);
    D(i) = f.D;
    a(i) = f.a;
%     figure(1);
%     plot(f, t_fit{i}(1:numFramesFit),d_fit{i}(1:numFramesFit));
%     pause()
end
close all
figure(1); 
hold on;
mean(D((a>0.9)&(a<1.2)))
nanmean(D((a>0.9)&(a<1.2)))
plot(a, D, 'b.');
% axis([0 2 0 1])

