function find_particles(logfile, fovn, calibrate, tracking)
%FIND_PARTICLES wraps core particle finding and tracking, using Kilfoil
    % fovn      - number of movie in folder
    % calibrate - whether function is used for calibration (only first frame)
    %             or for proper particle finding on the whole movie
    %             0: find all features, 1: only find in first frame
    % tracking  - 0: no tracking, 1: do tracking
    
    % Get number of frames from number of tiff files in directory
    cd(['fov', num2str(fovn)]);
    numFrames = length(dir('*.tif'));
    cd ..
    
    % Set parameters from logfile
    frame_interval = logfile.frame_interval(fovn);
    masscut = logfile.masscut(fovn);
    featsize = logfile.featsize(fovn);
    maxdisp = logfile.maxdisp(fovn);
    min_track_length = logfile.min_track_length(fovn);
    memory = logfile.memory(fovn);
    
    % Set parameters that are usually not changed
    basepath=[pwd, '/'];
    barint = 1;
    barrg = 50;
    barcc = 1;
    IdivRg = 0;
    Imin = 10;
    field = 2;
    frame = 1;
    
    % Save timesteps in format that mpretrack can read
    time = 0:frame_interval:numFrames*frame_interval;
    save(['fov', num2str(fovn), '_times.mat'], 'time');
    
    % Prerun, so parameters can be adjusted if necessary
    mpretrack_init( basepath, featsize, barint,...
        barrg, barcc, IdivRg, fovn, frame, masscut, Imin, field);
    drawnow;
    
    %Find features and track if running full analysis
    if ~calibrate
        mpretrack(basepath, fovn, featsize, barint, barrg, ...
                    barcc, IdivRg, numFrames, masscut, Imin, field );
    end
    % Link features if tracking == 1
    if ~calibrate && tracking
        fancytrack(basepath, fovn, featsize, maxdisp, min_track_length, memory );
    end
    
    % Clean up, if times.mat exists
    delete(['fov', num2str(fovn), '_times.mat']);