%c elegans project analysis pipeline

%% initialization
numFrames =1300; %number of frames to track
fovn = 2; %id of file
timestep = 0.033; % seconds per frame
fac=0.12; % in um per pixel

fps=1/timestep;
fname=['fov', num2str(fovn), '_tracks'];

%% Step 1: track particle and save track data in a data file
getparticletracks(timestep,numFrames,fovn)
portC2Arr(fname,fac)

%% Step 2: Plot all tracks and check track statistics
plottracks(fname)
tracksstats(fname)

%% Step 3: Plot MSD and make scatter plot 
msdArr=getMSD(fname,fps);
save(['msd' fname],'msdArr');
plotmsddist(fname);