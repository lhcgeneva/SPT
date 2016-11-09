%  Create 10 folders with identical simulation parameters, to test for
%  difference between python and Matlab code
for j = 1:10
mkdir(num2str(j));
cd(num2str(j));
%% Create trajectory of diffusing particle
pixel_size = 0.124;             % pixel size in µm
a = 27/pixel_size;              % half of long axis of the embryo in µm
b = 30/pixel_size;              % short axis of embryo in µm
species = [100, 100];
Ds = [0.2, 0.4] / pixel_size^2;   % mean diffusion constants
N = sum(species);                        % Number of molecules
D = [];
for i = 1 : length(species)           
    D = [D, Ds(i)*ones(1, species(i))];
end
var_coeff_det = 0.1;
var = var_coeff_det * D;                    % Variance of diffusion constant (in % of mean)         
% Diffusion rates for each reaction drawn from gaussian distribution with 
% mean d and variance a
d = var.*randn(1,N) + D; 
frame_interval = 0.033;                     % timestep size in s
timemax = 6;                                % in s
framemax = floor(timemax/frame_interval);   % Maximum number of timesteps
% Container for times and positions of molecules
allx = zeros(framemax,N);
ally = zeros(framemax,N);
% Initialize first frame with position of particles
allx(1,:) = a.*(rand(1,N) - ones(1,N)*0.5);
ally(1,:) = b.*(rand(1,N) - ones(1,N)*0.5);
% Simulate particle movement for the following frames
x_step = [];
y_step = [];
for n = 1:N
    for fnum = 1:framemax-1
        allx(fnum + 1, n) = allx(fnum, n) + ...
                            sqrt(abs(2 * frame_interval * d(1,n))) * randn(1);
        ally(fnum + 1, n) = ally(fnum, n) + ...
                            sqrt(abs(2 * frame_interval * d(1,n))) * randn(1);
    end
end
% Shift image origin
allx = allx - min(allx(:)) + 1;
ally = ally - min(ally(:)) + 1;
all_x_y = cat(3, allx, ally); % combine into one matrix
x_y_cell = cellfun(@(x) pixel_size*squeeze(x), mat2cell(all_x_y, framemax, ones(N, 1), 2),...
                    'UniformOutput', false);

%% Create parpool
parpool(4);
%% Create images
imageBaseValue = 950;
noiseVariance = 200;
Sigma = 1;                      % Spread of PSF in pixles
resolution = 0.1;               % Resolution within one pixel
aboveBG = 550;
tic
size_synthetic_movie = [ceil(max(allx(:))) + 3, ceil(max(ally(:))) + 3, framemax];
imMat = abs(noiseVariance*randn(size_synthetic_movie)+imageBaseValue);
parfor i = 1:size_synthetic_movie(3)
    [xall, yall, pall, im] = ...
                    sum_gaussians(size_synthetic_movie(1:2), resolution,...
                                  [allx(i, :)', ally(i, :)'], Sigma);
    imMat(:, :, i) = imMat(:, :, i) + aboveBG/mean(im(im>0)) * im;
end
toc
%% Write tifs and parameters used to create the data to disk
log_synth = table(frame_interval, pixel_size, a, b, N, Ds, var_coeff_det,...
                 imageBaseValue, noiseVariance, Sigma, resolution, aboveBG);
writetable(log_synth)
for i = 1:size_synthetic_movie(3)
    imwrite(uint16(imMat(:, :, i)),['fov12_', num2str(i,'%04d'), '.tif']);
end
cd ..
end