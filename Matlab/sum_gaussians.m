function [xall, yall, pall, imMat] = sum_gaussians(gridsize, resolution, partPos, Sigma)
%SUM_GAUSSIANS Creates image convolved with an artificial point spread function.
% PSF has breadth Sigma, within each pixel of image gridsize
sz = size(partPos);
xall = [];
yall = [];
pall = [];

% For each particle, create a meshgrid of size 2*Sigma around that particle, 
% link them all together and evaluate gaussian of spread Sigma and mean
% partPos at each point of the meshgrid.
for i = 1:sz(1)
    X = (partPos(i, 1)-2*Sigma:resolution:partPos(i, 1)+2*Sigma)';
    Y = (partPos(i, 2)-2*Sigma:resolution:partPos(i, 2)+2*Sigma)';
    [x, y] = meshgrid(X, Y);
    xall = [xall; x(:)];
    yall = [yall; y(:)]; 
    pall = [pall; mvnpdf([x(:) y(:)], partPos(i, :), Sigma)];
end

% Sort by ascending x-values
[xsorted, xindsSort] = sort(xall);
ysorted = yall(xindsSort);
psorted = pall(xindsSort);

% Initialize image matrix to project particles on
imMat = zeros(gridsize);

% Project particles onto image.
% Iterate over x dimension of image
for i = 1:gridsize(1)
    [temp_min, temp_max] = myFindDrGar(xsorted, i-1, i);
    % Iterate over y dimension of image
    for j = 1:gridsize(2)
        if ~isempty(temp_min) && ~isempty(temp_max)
            ytemp = ysorted(temp_min:temp_max);
            ptemp = psorted(temp_min:temp_max);
            % Sum all values from pall that have an x and y value within
            % the current pixel (i, j).
            imMat(i, j) = sum(ptemp(ytemp>j-1 & ytemp<=j));
        end
    end
end
      