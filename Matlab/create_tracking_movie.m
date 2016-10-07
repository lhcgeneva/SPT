function create_tracking_movie(numFrames, fovn)
%CREATE_TRACKING_MOVIE Make movie with tracked particles overlayed
%   CREATE_TRACKING_MOVIE(numFrames, fovn)
%           numFrames - First numFrames frames from movie are taken
%           fovn      - number of movie in directory

% Read images
cd(['fov', num2str(fovn)])
im = imread(['fov', num2str(fovn), '_0001.tif']);
im = zeros([numFrames, size(im)]);
for i = 1:numFrames
    im(i, :, :) = imread(['fov', num2str(fovn), '_', ...
                            num2str(i,'%04d'), '.tif']);
end
cd ..

% Display movie
cd('Bead_tracking/res_files/');
load(['res_fov', num2str(fovn), '.mat']);
cd ../..
v = VideoWriter(['Tracking_50p_', num2str(fovn), '.avi']);
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