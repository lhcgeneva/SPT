% Run init_params.m (my code) and mainscript.m (Tzer's code) on the same
% movie (in this case I used 
% /Users/hubatsl/Desktop/SPT/Us/SPT/sample_data/16_07_20_PAR6_2/fov1) to
% show that Tzer's and my code are identical except for our handling of NaNs.
% This becomes evident when he stores tau in the structure msdArr in
% lines 15 and 44 of file getMSD.m. He takes positions from the output
% array of the Kilfoil array, which does not contain nan's for positions
% not found, but instead records the time when the image was created. He
% than replaces time by a tau which is simply interpolated between start
% and end determined by length(msd). This gives artificially high jumps for
% points that are not tracked within a track.
% Leaving nans should be more precise, as I'm not creating artificially
% large jumps. 

% To redo this, save the above images in fov1 in the current folder (8bit!),
% make a copy of the same folder called fov2, with the images in it also
% called fov2... instead of fov1... Tzer's code will be run on fov2, mine
% will be run on fov1.

% Then run the init_params.m and mainscript.m with the parameter set used in 
% the first commit containing this file and do the following:

% The two following plots should be on top of each other, as track number
% 56 in this movies using these parameters does not contain any nan values
load fov2_tracks % load Tzer's tracks
plot(x_y_cell{56}(:, 1)/0.12, x_y_cell{56}(:, 2)/0.12); hold on;
plot(c{56}(:, 1), c{56}(:, 2)); 
msdArr(56).msd==msd{56}(2:end-9) % Tzer's msds are shorter because he seems
                                 % to cut off the last couple of lag times

load msdfov2_tracks % load Tzer's msd calculated
for i=1:length(msdArr)
    msdi=msdArr(i);
    pB=msdi.pB;
    vAlpha(i)=pB(1);
    vD(i)=(10^pB(2))/4;
end

% The following two plots should be similar but not the same, due to the
% nan issue mentioned above.
figure; hold on;
scatter(a,D)
scatter(vAlpha, vD) 

% There does not seem to be an obvious correlation between how many nan
% values there are per track and how its position in the a vs D graph:
figure;
weights_unnorm = cell2mat(cellfun(@(x) sum(sum(isnan(x))),...
                          x_y_cell, 'UniformOutput', 0));
scatter(a, D, 25, weights_unnorm, 'filled');
% Now the same normalized to tracklength:
figure;
weights = cell2mat(cellfun(@(x) sum(sum(isnan(x))), x_y_cell,... 
                           'UniformOutput', 0))./cellfun(@length, x_y_cell);
scatter(a, D, 25, weights, 'filled');

% Compare Ds: His are slightly larger than mine, presumable due to the
% above mentioned issue with nans:
figure; hold on;
histogram(vD,[0:0.05:0.7])
histogram(D,[0:0.05:0.7])
legend('Tzer', 'Lars')

mean(vD)
mean(D)