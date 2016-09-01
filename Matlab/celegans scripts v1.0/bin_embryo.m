function [posArrRot, trackBinNum] = bin_embryo(numBins, folderName, ...
                                               posArr, pixelSize)
%BIN_EMBRYO Rotates tracks so anterior is on the left, bins tracks
%       Orientation of original embryo is obtained by clicking on anterior
%       and posterior (order matters!). Output posArrRot contains rotated x
%       and y positions (just like posArr) and the bin number for each
%       track.

%% Display figure, click on anterior/posterior
f = figure();
im = imread([folderName, '/', folderName, '_0001.tif']);
h = fspecial('gaussian', [10 10], 30);
blurred = imfilter(im,h,'replicate');
imagesc(blurred); shg; hold on;
title('Click on anterior (first) and posterior (second) poles.');
axis equal;
[X, Y] = ginput(2);
X = X * pixelSize;
Y = -Y * pixelSize;

%% Get angle between x-axis and embryo axis (0...360), create rotation matrix
ydiff = Y(2)-Y(1);
xdiff = X(2)-X(1);
angle = acos(xdiff/(norm([xdiff, ydiff])))*180/pi;
if ydiff > 0; angle = -angle; end;
R = rotz(angle);
axesRotated1 = R * [X(1); Y(1); 0];
axesRotated2 = R * [X(2); Y(2); 0];
edges = linspace(axesRotated1(1), axesRotated2(1), numBins + 1);

%% Rotate every track in struct posArr, store in output struct posArrRot
posArrRot(length(posArr)).xRot = [];
posArrRot(length(posArr)).yRot = [];
for i = 1:length(posArr)
    temp3xN = [posArr(i).x'; posArr(i).y'; zeros(1, length(posArr(i).x))];
    temp3xNRot = R * temp3xN;
    posArrRot(i).xRot = temp3xNRot(1, :);
    posArrRot(i).yRot = temp3xNRot(2, :);
    posArrRot(i).avXRot  = mean(posArrRot(i).xRot);
end  
trackBinNum = discretize([posArrRot.avXRot], edges);
%% Plots 
% Plot original orientation with axis 
g = figure(); hold on;
for i = 1:length(posArr)
    plot(posArr(i).x, posArr(i).y);
end

plot([X(1), X(2)], [Y(1), Y(2)], '-');
% Plot final orientation with axis and histogram of the number of tracks
h = figure(); hold on;
for i = 1:length(posArr)
    plot(posArrRot(i).xRot, posArrRot(i).yRot);
    plot(posArrRot(i).avXRot, zeros(length(posArrRot(i).avXRot), 1), 'r.', ...
         'MarkerSize', 15);
end
plot([axesRotated1(1), axesRotated2(1)], [axesRotated1(2), axesRotated2(2)], ...
        '-');
histogram([posArrRot.avXRot], edges);

% Checking whether discretize and histogram agree
l = figure();
histogram(trackBinNum);

% Pause until user keyboard input, close all figures
pause()
try 
    close([f, g, h, l]);
catch 
    close all;
end
    
   