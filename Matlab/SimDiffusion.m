
%% Create trajectory of diffusing particle
clear all;

% long and short axis of embryo in µm
a = 27;
b = 15;

% Number of molecules
N = 200;

% mean diffusion constant
D = 0.15;
% Variance of diffusion constant
var = 0.05;           
d = var.*randn(1,N) + D; % Diffusion rates for each reaction drawn from gaussian distribution with mean d and variance a

timemax = 40;
dt = 0.04;      % timestep size
mult = 1;
tau = mult*dt;
framemax = floor(timemax/tau);   % Maximal number of timesteps
noise = 0.5; % Strength of background noise
pixelSize = 0.1049
% Container for times and positions of
% molecules
allx = zeros(framemax,N);
ally = zeros(framemax,N);

allx(1,:) = 2*a.*(rand(1,N) - ones(1,N)*0.5);
ally(1,:) = 2*b.*(rand(1,N) - ones(1,N)*0.5);

for n = 1:N
    for fnum = 1:framemax-1
        allx(fnum+1,n) = allx(fnum,n) + (4*tau*d(1,n)).*randn(1);
        ally(fnum+1,n) = ally(fnum,n) + (4*tau*d(1,n)).*randn(1);
    end
end

%% Plotting
% +1 in definition of newx and new y to avoid error when indexing in matInt
% (alogithm uses minimum as index for array, so 0 doesn't work)
newx = round((allx - min(allx(:)))/pixelSize+1);
newy = round((ally - min(ally(:)))/pixelSize+1);
figure()
plot(newx(:,1:end),newy(:,1:end),'.')
%%
allI = zeros(1,N);
allI(1,:) = rand(1,N) + 10*ones(1,N)*0.5;

matInt = zeros(framemax,ceil(max(newx(:))), ceil(max(newy(:))));
matImGauss = zeros(framemax,ceil(max(newx(:))), ceil(max(newy(:))));
matFull = zeros(framemax,ceil(max(newx(:))), ceil(max(newy(:))));
%%
for f = 1:framemax
    for n = 1:N
        matInt(f,newx(f,n),newy(f,n)) = allI(1,n);
    end
    matImGauss(f,:,:) = imgaussfilt3(matInt(f,:,:),2);
    matIm = noise*rand(1,ceil(max(newx(:))), ceil(max(newy(:))));
    matFull(f,:,:) = matIm + matImGauss(f,:,:);
end
%%
fovn = 1;
for i = 1:1000
    figure('Visible', 'Off')
    imshow(squeeze(matFull(i,:,:)))
    imwrite(squeeze(matFull(i,:,:)), ['fov', num2str(fovn), '_', num2str(i, '%04d'), '.tif']);
end

