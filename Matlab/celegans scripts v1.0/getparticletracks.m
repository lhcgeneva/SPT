function getparticletracks(timestep,numFrames,fovn)

% numFrames =200;
% timestep = 0.033;
% fovn = 1;

time = 0:timestep:numFrames*timestep;
save(['fov', num2str(fovn), '_times.mat'], 'time');

pixelSize = 0.1049;
numFramesFit = 10;
basepath=[pwd, '/'];
featsize = 3;
barint = 1;
barrg = 50;
barcc = 1;
IdivRg = 0;
Imin = 10;
masscut = 230;
field = 2;
frame = 1;
% Prerun, so parameters can be adjusted if necessary
[M2, MT] = mpretrack_init( basepath, featsize, barint,...
    barrg, barcc, IdivRg, fovn, frame, masscut, Imin, field);
drawnow;
% delete(['fov', num2str(fovn), '_times.mat']);
pause()
% Find features
% Save timesteps in format that mpretrack can read
disp('-------------------------------------------')
disp('Running find features ...')

% Do the tracking
mpretrack(basepath, fovn, featsize, barint, barrg, ...
            barcc, IdivRg, numFrames, masscut, Imin, field );
% Clean up
delete(['fov', num2str(fovn), '_times.mat']);
disp('Feature finding done!')

% %%%%%%%%%%%%%%%%%%%%%%% Diffusion rate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
% Link features
disp('-------------------------------------------')
disp('Linking features ...')
maxdisp = 5;
min_track_length = 80;
memory = 7;
fancytrack(basepath, fovn, featsize, maxdisp, min_track_length, memory );
disp('Feature linking done!')

% Plot histogram of track lengths longer than min_track_length
figure(1); clf;
cd Bead_tracking/res_files;
load(['res_fov', num2str(fovn), '.mat']);
cd ../..
c = cell(1, max(res(:, 8)));
for i = 1:max(res(:, 8))
    c{i} = res(res(:, end) == i, :);
end

save(['fov', num2str(fovn), '_tracks.mat'],'c')