# Author: Lars Hubatsch, object oriented wrapper for trackpy
from IPython.core.debugger import Tracer
from itertools import repeat, product
from matplotlib import cm, colors
from matplotlib.path import Path
from matplotlib.pyplot import (axis, close, figure, gca, gcf, get_cmap, hist,
                               imshow, ioff, savefig, scatter, show, subplots,
                               setp)
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from multiprocessing import Pool
from numpy import (arange, array, asarray, c_, ceil, cumsum, diag, diff, dot,
                   exp, histogram, invert, linspace, log10, mean, polyfit,
                   random, round, searchsorted, select, shape, sqrt, sum,
                   shape, transpose, zeros)
from pandas import concat, DataFrame
from pickle import dump, HIGHEST_PROTOCOL
from pims import ImageSequence
from pims_nd2 import ND2_Reader
from os import chdir, makedirs, path, stat
from re import findall, split
from scipy import integrate, optimize
from shutil import rmtree
from sys import exit
from tifffile import imsave, TiffFile
import warnings

from thirdParty.roipoly import roipoly  # In Jupyter set %matplotlib notebook
import seaborn as sns
import trackpy as tp

'''
TO DO:
Separate arguments for DiffusionFitter/OffRateFitter/ParticleFinder
Fix file reading, there's a lot of redundancy (.stk files are read
but not written in write_images, etc...)
'''


class ParticleFinder(object):

    '''
    Wraps trackpy so properties of individual movies can be conveniently
    accessed through class interface. Every movie becomes one instance of the
    class. ParticleFinder is a generic interface, for off rates or
    diffusion rates use OffRateFitter or DiffusionFitter

    Parameters:
    filePath            Path of .stk file or folder containing tiffs
    threshold           minmass parameter for trackpy
    autoMetaDataExtract Boolean, decide whether to extract meta data from file
    dist                search_range parameter for trackpy, distance a particle
                        can move between frames in pixels
    featSize            diameter parameter for trackpy, approx. feature size
    maxsize             maxsize parameter for trackpy
    memory              memory parameter for trackpy
    minTrackLength      Minimum track length for trackpy (func filter_stubs)
    no_workers          Number of parallel processes to be started if parallel
                        is true
    parallel            Boolean, to run feature finding on more than one core
    pixelSize           Pixel size of microscope in microns
    saveFigs            Boolean, to save figures or not to save
    showFigs            Boolean, to show figures after creating them
    startFrame          First frame to take into consideration for analysis
                        for off rates
    timestep            real time difference between frames
    '''

    def __init__(self, filePath=None, threshold=40, adaptive_stop=None,
                 autoMetaDataExtract=True, dist=5, endFrame=None, featSize=7,
                 link_strat='drop', maxsize=None, memory=7, minTrackLength=80,
                 no_workers=8, numFrames=10, parallel=True, pixelSize=0.124,
                 saveFigs=False, showFigs=False, startFrame=0, startLag=1,
                 timestep=None):
        '''
        Initialize ParticleFinder object. Make sure to check whether all
        standard parameters make sense, in particular dist, featSize,
        link_strat, pixelSize, memory, minTrackLength
        '''
        self.no_workers = no_workers
        self.parallel = parallel
        self.saveFigs = saveFigs
        self.showFigs = showFigs
        self.timestep = timestep
        if filePath is not None:
            self.adaptive_stop = adaptive_stop
            self.autoMetaDataExtract = autoMetaDataExtract
            self.dist = dist
            self.endFrame = endFrame
            self.featSize = featSize
            self.link_strat = link_strat
            self.maxsize = maxsize
            self.memory = memory
            self.minTrackLength = minTrackLength
            self.numFrames = numFrames
            self.pixelSize = pixelSize
            self.startFrame = startFrame
            self.startLag = startLag
            self.stackPath = filePath
            self.basePath = split('/', self.stackPath[::-1], 1)[1][::-1] + '/'
            self.stackName = self.stackPath.split('/')[-1].split('.')[0]
            self.threshold = threshold
            self.load_frames()

    def append_output_to_csv(self, csv_path, data):
        cols = [a for a in data.keys()]
        df = DataFrame(data, index=[0])
        with open(csv_path, 'a') as f:
            # Check whether file empty, if not omit header
            # The columns need to be in alphabetical order, because of pandas
            # bug, should be fixed in next pandas release.
            if stat(csv_path).st_size == 0:
                # Make sure to keep alphabetical order!!!
                df.to_csv(f, header=True, columns=cols)
            else:
                df.to_csv(f, header=False, columns=cols)

    def apply_ROI(self, useAllFeats=False, inversion=False):
        '''
        Filter all found features by whether they have been found
        within this self.ROI
        
        useAllFeats     get features by masking out from features_all
        inversion       inverts mask, so all particles outside the mask are
                        used for quantification
        '''
        if useAllFeats:
            features = self.features_all
        else:
            features = self.features
        bbPath = Path(asarray(
                    list(zip(*(self.ROI.allxpoints, self.ROI.allypoints)))))
        x_y_tuples = list(zip(*(features['x'].values,
                                features['y'].values)))
        mask = [bbPath.contains_point(asarray(i)) for i in x_y_tuples]
        if inversion:
            self.features = features[invert(mask)]
        else:
            self.features = features[mask]
        self.partCount, _ = histogram(self.features.frame,
                                      bins=self.features.frame.max()+1)

    def def_ROI(self, n=0):
        '''
        Define a ROI in the nth frame,
        '''
        imshow(self.frames[n])
        self.ROI = roipoly(roicolor='r')

    def delete_images(self):
        '''
        Delete images from tif directory.
        '''
        if '.stk' in self.stackPath:
            tif_path = self.basePath+self.stackName
            if path.exists(tif_path):
                rmtree(tif_path)
        else:
            print('Not deleting files, no .stk file available ' +
                  'tiffs are potentially the only record of the data')

    def find_feats(self):
        '''
        Check whether parallel execution is on, otherwise do normal batch
        processing
        '''
        if self.showFigs:
            self.plot_calibration()

        if self.parallel:
            # Create list of frames to be analysed by separate processes
            self.f_list = []
            # Get size of chunks
            s = ceil(len(self.frames) / self.no_workers)
            for ii in range(0, self.no_workers - 1):
                # Issue with index, check output!
                self.f_list.append(self.frames[int(s * ii):int(s * (ii + 1))])
            # Last worker gets slightly more frames
            self.f_list.append(self.frames[int(s * (self.no_workers - 1)):])
            # Create pool, use starmap to pass more than one parameter, do work
            pool = Pool(processes=self.no_workers)
            res = pool.starmap(tp.batch, zip(self.f_list,
                                             repeat(self.featSize),
                                             repeat(self.threshold),
                                             repeat(self.maxsize)))
            # Concatenate results and assign to features (this is what every
            # other function works on, except apply_ROI()) and features_all,
            # which is kept to always be able to go back to working on the full
            # set of points
            self.features_all = concat(res)
            self.features = self.features_all
            # Close and join pool
            pool.close()
            pool.join()
            self.f_list = []
        else:
            self.features_all = tp.batch(self.frames[:], self.featSize,
                                         minmass=self.threshold,
                                         maxsize=self.maxsize,
                                         invert=False)
            self.features = self.features_all

    def load_frames(self):
        '''
        Load data from path, check whether to extract frame intervals
        automatically
        '''
        if self.autoMetaDataExtract and '.nd2' in self.stackPath:
            frames = ND2_Reader(self.stackPath)
            self.timestep = (frames[-1].metadata.get('t_ms', None) /
                             (len(frames)-1))
            frames.close()
        elif self.autoMetaDataExtract and '.stk' in self.stackPath:
            tif = TiffFile(self.stackPath)
            string = tif.pages[0].image_description[0:1000].decode("utf-8")
            idx1 = string.find('Exp')
            idx2 = string.find('ms')
            exp_string = string[idx1:idx2+4]
            self.exposure = float(findall('\d+', exp_string)[0])/1000
            time_created = tif.pages[0].uic2tag.time_created/1000  # in sec
            self.timestep = mean(diff(time_created))
            if self.timestep is not None:
                print('Input time step ignored, extracted timestep is ' +
                      str(self.timestep))
        elif ((self.autoMetaDataExtract and '.nd2' not in self.stackPath) or
              self.timestep is None):
            print(''''Metadata extraction currently only supported for
                       .nd2 and .stk files. Please provide the timestep as
                       optional argument to ParticleFinder.''')
        # If file format of stack is .nd2 read and write stack
        if 'nd2' or '.stk' or '.tif' in self.stackPath:
            self.write_images()
        # Read in image sequence from newly created file
        self.frames = ImageSequence(self.basePath+self.stackName+'/*.tif',
                                    as_grey=True)
        # Careful, startFrame is for now only supported in DiffusionFitter,
        # not in OffRateFitter, leave at zero for off rates until the fit
        # methods have been verified to work with a startFrame different from 0
        if self.endFrame is None:
            self.frames = self.frames[self.startFrame:]
        else:
            self.frames = self.frames[self.startFrame:self.endFrame]

    def pickle_data(self, path=None, postfix=''):
        '''
        Pickle data to store on disk.

        path     save pickled object elsewhere other than in the directory
                 containing the imaging data.
        postfix  add a postfix to the filename
        '''
        if path is None:
            path = self.basePath + self.stackName + '.pickle'
        else:
            path = path + self.stackName + postfix + '.pickle'
        self.frames = []  # Do not store image data (save speed and resources)
        with open(path, 'wb') as f:
            # Pickle self using the highest protocol available.
            dump(self, f, HIGHEST_PROTOCOL)

    def plot_calibration(self, calibrationFrame=0):
        self.set_fig_style()
        imshow(self.frames[calibrationFrame])
        f = tp.locate(self.frames[calibrationFrame], self.featSize,
                      invert=False, minmass=self.threshold,
                      maxsize=self.maxsize)
        fig = tp.annotate(f, self.frames[calibrationFrame])
        if self.saveFigs:
            fig1 = fig.get_figure()
            fig1.savefig(self.stackPath + 'Particle_Calibration' +
                         str(calibrationFrame) + '.png',
                         bbox_inches='tight')

    def save_summary_input(self):
        data = {'dist': self.dist, 'featSize': self.featSize,
                'memory': self.memory, 'no_workers': self.no_workers,
                'parallel': self.parallel, 'pixelSize': self.pixelSize,
                'timestep': self.threshold, 'timestep': self.timestep,
                'path': self.stackPath}
        frame_to_write = DataFrame(data, index=[0])
        frame_to_write.to_csv(self.basePath + 'summary'+self.stackName+'.csv')
        print('Summary saved.')

    def set_fig_style(self):
        ioff()
        sns.set_context("poster")
        sns.set_style("dark")

    def write_images(self):
        if not path.exists(self.basePath+self.stackName):
            makedirs(self.basePath+self.stackName)
            dir_path = path.dirname(path.realpath(__file__))
            chdir(self.basePath+self.stackName)
            if '.stk' or '.tif' in self.stackPath:
                frames = TiffFile(self.stackPath).asarray()
            else:
                frames = ND2_Reader(self.stackPath)
            if len(frames.shape) == 2:
                imsave(self.stackName+'.tif', frames)
            else:
                for i in range(len(frames)):
                    imsave(self.stackName+'_'+str(i)+'.tif', frames[i])
            chdir(dir_path)
        else:
            print('Path already exists, not writing.')


class DiffusionFitter(ParticleFinder):

    '''
    Example call for debugging:
    d = DiffusionFitter('/Users/hubatsl/Desktop/DataSantaBarbara/
                        Aug_09_10_Test_SPT/th411_P0_40ms_100p_299gain_1678ang_
                        earlyMaintenance.nd2', 1100, autoMetaDataExtract=True)

    Extends ParticleFinder to implement mobility calculation
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def analyze(self, analyzeTracks=True):
        '''
        This should only be run the first time a movie is analyzed,
        as it uses features_all.
        '''
        super().find_feats()
        # To be able to check whether frame has been assigned in link_feats()
        # assign to empty data frame first
        self.trajectories = DataFrame([])
        self.im = DataFrame([])
        self.trajectories_all = DataFrame([])
        self.im_all = DataFrame([])
        self.link_feats(useAllFeats=True)
        if analyzeTracks:
            self.analyze_tracks()
        if self.showFigs:
            self.plot_trajectories()
            self.plot_msd()
            self.plot_diffusion_vs_alpha()
        self.save_output()

    def analyze_tracks(self):
        # Convert to numpy, get diffusion coefficients
        numParticles = self.trajectories['particle'].nunique()
        imAsNumpyArray = self.im.as_matrix()
        self.time = linspace(self.startLag*self.timestep,
                             self.timestep*(self.startLag+self.numFrames-1),
                             self.numFrames)
        DA = zeros([numParticles, 2])
        self.res = zeros([numParticles, 1])
        for j in range(0, numParticles):
            MSD = imAsNumpyArray[self.startLag-1:self.startLag +
                                 self.numFrames-1, j]
            results = polyfit(log10(self.time), log10(MSD), 1, full=True)
            DA[j, ] = [results[0][0], results[0][1]]
            self.res[j] = results[1][0]
        self.D = 10**DA[:, 1]/4
        self.a = DA[:, 0]
        self.D_restricted = mean(self.D[(self.a > 0.9) & (self.a < 1.2)])

    def filt_traj_by_t(self, start, stop):
        '''
        Trajectories are filtered by time only leaving trajectories whose mean
        frame number is greater or equal to start and smaller or equal to stop.
        Start and stop take the total length of the movie as a reference, 
        similar to startFrame and endFrame.
        If startFrame==800 and endFrame=1500, start and stop could be
        900 and 1100, for example.
        '''
        self.trajectories = self.trajectories.groupby('particle').filter(
            lambda p: (mean(p['frame']) >= start) &
            (mean(p['frame']) <= stop))

    def make_labelled_movie(self, parts, isolate=True):
        '''
        Make movie with particles in list [parts] labelled with black box
        '''
        a = 4  # Distance of rectangle from particle midpoint
        fs = array(self.frames)
        for part in parts:
            # Get particle trajectory and its start and end frame
            p = self.trajectories.groupby('particle').get_group(part)
            start = min(p.frame) - self.startFrame
            stop = max(p.frame) - self.startFrame + 1
            # Draw black rectangle around particle
            x = p.x.round().astype(int)
            y = p.y.round().astype(int)
            for j in range(len(fs)):
                if (j >= start) & (j < stop):
                    i = self.startFrame+j
                    if i in y.index:
                        fs[j, y.loc[i]-a:y.loc[i]+a, x.loc[i]-a] = 0
                        fs[j, y.loc[i]-a:y.loc[i]+a, x.loc[i]+a] = 0
                        fs[j, y.loc[i]-a, x.loc[i]-a:x.loc[i]+a] = 0
                        fs[j, y.loc[i]+a, x.loc[i]-a:x.loc[i]+a] = 0
        if isolate:
            context = 30  # At least 30 pixels distance on all sides
            for part in parts:
                p = self.trajectories.groupby('particle').get_group(part)
                start = min(p.frame) - self.startFrame
                stop = max(p.frame) - self.startFrame + 1
                pos = round(array([min(p.x) - context, max(p.x) + context,
                            min(p.y) - context, max(p.y) +
                            context])).astype(int)
                pos[pos < 0] = 0
                imsave(self.basePath+str(int(part)) + '.tif',
                       fs[start:stop, pos[2]:pos[3], pos[0]:pos[1]])
        else:
            imsave(self.basePath+'_'.join(self.stackPath.split('/')[-2:]) +
                   'annotated'+'.tif', fs)

    def hist_step_size(self, histCut=None, n=1, numBin=None):
        '''
        Plot histogram of step sizes.

        histCut     cuts off the histogram, defaults to self.dist
        n           number of frames to be jumped.
        numBin      number of bins in histogram
        '''
        if histCut is None:
            histCut = self.dist
        g = self.trajectories.groupby('particle')
        h = g.apply(lambda p: sqrt((p['x'][n:].values-p['x'][:-n].values)**2 +
                                   (p['y'][n:].values-p['y'][:-n])**2)).values
        # apply().values seems to give different format, if groupby only has
        # one group, therefore treat this case separately below.
        if len(g) == 1:
            h = h[0]
        h_cut = [x for x in h if x < histCut]
        if numBin is None:
            hist(h_cut)
        else:
            hist(h_cut, numBin)
        show()
        return h_cut

    def link_feats(self, useAllFeats=False, mTL=None):
        '''
        Link individual frames to build trajectories, filter out stubs shorter
        than minTrackLength. Get Mean Square Displacement (msd).

        useAllFeats     True when linking should be run on all features,
                        False when linking should be run only on the subset
                        self.features, e.g. after apllying a ROI
        '''

        # For better readability define all parameters here
        dist = self.dist
        memory = self.memory
        pixelSize = self.pixelSize
        timestep = self.timestep
        if mTL is None:
            mTL = self.minTrackLength
        if useAllFeats:
            features = self.features_all
        else:
            features = self.features

        # Run linking using the above parameters
        # Can be run with diagnostics=True to get diagnostics, see docs.
        # link_strategy='drop' drops particle instead of resolving subnetwork.
        t = tp.link_df(features, dist, memory=memory,
                       link_strategy=self.link_strat,
                       adaptive_stop=self.adaptive_stop)
        trajectories = tp.filter_stubs(t, mTL)
        # Get msd
        im = tp.imsd(trajectories, pixelSize, 1 / timestep)

        # Assign to the object attributes specified by useAllFeats
        if useAllFeats:
            self.trajectories_all = trajectories
            self.im_all = im
        else:
            self.trajectories = trajectories
            self.im = im

        # If link_feats hasn't been run before, self.trajectories and self.im
        # are empty, therefore assign trajectories and im to self.trajectories
        # and self.im, so they are the same as trajectories_all and im_all
        if (self.trajectories.empty and self.im.empty):
            self.trajectories = trajectories
            self.im = im

    def plot_diffusion_vs_alpha(self, xlim=None, ylim=None, percentile=0.9):
        '''
        Plots diffusion coefficient vs anomalous diffusion exponent
        xlim        range for x values to be shown
        ylim        range for y values to be shown
        percentile  sets upper limit of dynamic range used for colorcode
        '''
        grouped = self.trajectories.groupby('particle')

        # Decide what information goes into color code
        # Particle mass
        weights = [mean(group.mass) for name, group in grouped]
        # Number of frames particle was found in
        weights = self.trajectories.particle.value_counts(sort=False).tolist()
        # Residues of fit
        weights = [x[0] for x in self.res]
        # # Appearance time relative to movie length
        # weights = (self.trajectories.groupby('particle').
        #            apply(lambda p: mean(p['frame'])))
        # if not weights.index.is_monotonic:
        #     print('Aborting, time might not be grouped with right D/a.')
        #     exit()
        # weights = weights.values
        # Get histogram of weights to choose a sensible dynamic range for
        # the color code in case the weights have big top outliers
        h = histogram(weights, 20)  # 20 bins
        cs_h = cumsum(h[0])
        ma = max(cs_h)  # Get maximum of cumulative sum
        perc = ma * percentile  # Get 90th percentile
        idx = searchsorted(cs_h, perc)  # Find index in histogram bins
        thresh = h[1][idx]  # Get threshold from histogram bins
        cs_thresh = [x if x <= thresh else thresh for x in weights]

        # Build color map
        autumn = get_cmap('winter_r')  # Set colormap
        cNorm = colors.Normalize(vmin=min(cs_thresh), vmax=max(cs_thresh))
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=autumn)
        cs_scaled = [scalarMap.to_rgba(x) for x in cs_thresh]

        # Build figure
        self.set_fig_style()
        fig = figure()
        ax = fig.add_subplot(111)
        # Histogram inset
        inset_hist = inset_axes(ax, width="35%", height="35%", loc=2,
                                borderpad=3.)
        inset_hist.title.set_text('Weight hist and cut-off used for coloring.')
        inset_hist.title.set_size(10)
        # Color histogram according to colorcode
        n, bins, patches = inset_hist.hist(weights, 20)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])[:idx]
        patches = patches[:idx]
        ids = searchsorted(cs_thresh, bin_centers)
        cNorm = colors.Normalize(vmin=min(bin_centers), vmax=max(bin_centers))
        scalarMap1 = cm.ScalarMappable(norm=cNorm, cmap=autumn)
        cs_scaled1 = [scalarMap.to_rgba(x) for x in bin_centers]
        for c, p in zip(cs_scaled1, patches):
            setp(p, 'facecolor', c)
        inset_hist.plot((thresh, thresh), (0, max(h[0])), 'k-')
        inset_hist.tick_params(labelsize=9)
        # Scatter plot alpha vs D
        ax.scatter(self.a, self.D, c=cs_scaled, edgecolors='none', alpha=0.5,
                   s=50)
        ax.set(ylabel='D [$\mu$m$^2$/$s$]', xlabel=r'$\alpha$')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        # Save and show figure
        if self.saveFigs:
            savefig(self.stackPath + 'Particle_D_a.pdf', bbox_inches='tight')
        if self.showFigs:
            show()
        close()

    def plot_diffusion_vs_alpha_verbose(self):
        df = DataFrame({'D': self.D, 'a': self.a})
        sns.set(style="white")
        sns.set(font_scale=2)
        g = sns.PairGrid(df, diag_sharey=False, size=8)
        g.map_lower(sns.kdeplot, cmap="Blues_d")
        g.map_upper(scatter)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g.map_diag(sns.kdeplot, lw=int(3))
        fig = gcf()
        if self.saveFigs:
            fig.savefig(self.stackPath + 'Diff_vs_A_verbose' + '.pdf',
                        bbox_inches='tight')
        show()

    def plot_trajectories(self, label=False):
        self.set_fig_style()
        ax = gca()
        axis('equal')
        fig = tp.plot_traj(self.trajectories, label=label)
        fig = fig.get_figure()
        if self.saveFigs:
            fig.savefig(self.stackPath + 'Particle_Trajectories' + '.pdf',
                        bbox_inches='tight')
        close()

    def plot_msd(self):
        self.set_fig_style()
        fig, ax = subplots()
        fig.suptitle('MSD vs lag time', fontsize=20)
        ax.plot(self.im.index, self.im, 'k-', alpha=0.4)  # already in sec
        ax.set(ylabel='$\Delta$ $r^2$ [$\mu$m$^2$]', xlabel='lag time $t$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        if self.saveFigs:
            savefig(self.stackPath + 'Particle_msd.pdf', bbox_inches='tight')
        if self.showFigs:
            show()
        close()

    def save_output(self):
        super().save_summary_input()
        # Check whether a and D have been created
        if hasattr(self, 'a') and hasattr(self, 'D'):
            columns = ['a', 'D']
            combinedNumpyArray = c_[self.a, self.D]
            d = DataFrame(data=combinedNumpyArray, columns=columns)
            d.to_csv(self.basePath + 'D_a_'+self.stackName+'.csv')
        else:
            print('D and a are not available.')

    def gData(self):
        return {'Alpha_mean': self.a.mean(), 'D_mean': self.D.mean(),
                'D_restr': self.D_restricted, 'File': self.stackPath}

    def velocity_measurements(self):

        df = self.trajectories

        # x and y coordinate differences for each frame for each particle
        df['xdiff'] =  df.groupby('particle')['x'].apply(lambda x: x - x.iloc[0])
        df['ydiff'] =  df.groupby('particle')['y'].apply(lambda y: y - y.iloc[0])

        # calculating the max displacement using Pythagorus (and adusting from pixels to microns)
        xdiff_sq =  df.groupby('particle')['xdiff'].apply(lambda x: x ** 2)
        ydiff_sq =  df.groupby('particle')['ydiff'].apply(lambda y: y ** 2)
        df['disp'] =  ((xdiff_sq + ydiff_sq) ** 0.5) * self.pixelSize

        # calculating number of frames at which particle achieves max displacement
        df['frame_diff'] =  df.groupby('particle')['frame'].apply(lambda x: x - x.iloc[0])

        # finding the final displacement
        final_d = df.groupby('particle')['disp'].apply(lambda x: x.iloc[-1])
        frames_at_final_d = df.groupby('particle')['frame_diff'].apply(lambda x: x.iloc[-1])

        # finding maximum displacement (NOT final displacement)
        idx = df.groupby(['particle'])['disp'].transform(max) == df['disp']
        df2 = df[idx]
        df2 = df2.copy()

        # deleting unnnecesary columns and renaming
        df2.drop(df2.columns[[0,1,2,3,4,5,6,7]], axis=1, inplace=True)
        df2.rename(columns={'disp': 'max_disp', 'frame_diff':'frames_at_max'}, inplace=True)

        # direction of movement at maximum displacement
        conditions = [
            (df2['xdiff'] > 0 ) & (df2['ydiff'] > 0 ),
            (df2['xdiff'] > 0 ) & (df2['ydiff'] < 0 ),
            (df2['xdiff'] < 0 ) & (df2['ydiff'] > 0 ),
            (df2['xdiff'] < 0 ) & (df2['ydiff'] < 0 )
        ]
        directions = ['Right upwards','Right downwards','Left upwards','Left downwards']
        df2['direction_at_max'] = select(conditions, directions)

        # calculating max velocity 
        df2['velocity_at_max'] = (df2['max_disp']/ (df2['frames_at_max']*(self.timestep/60)))

        # calculating final velocity
        df2['final_disp'] = final_d.values
        df2['frames_at_final'] = frames_at_final_d.values
        df2['velocity_at_final'] = (df2['final_disp']/ (df2['frames_at_final']*(self.timestep/60)))

        # exporting to csv
        df2.to_csv(self.stackPath[:-4] + '_velocities.csv')

class OffRateFitter(ParticleFinder):

    '''
    Extends ParticleFinder to implement Off rate calculation
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def analyze(self):
        super().find_feats()
        self.partCount, _ = histogram(self.features.frame,
                                      bins=self.features.frame.max()+1)
        if self.autoMetaDataExtract and '.stk' in self.stackPath:
            times_created = (TiffFile(self.stackPath).pages[0].
                             uic2tag.time_created/1000)  # in seconds
            self.fitTimes = times_created-times_created[0]
        else:
            self.fitTimes = arange(0, len(self.partCount)*self.timestep,
                                   self.timestep)

    def fit_offRate(self, variants=[1, 2, 3, 4, 5, 6]):
        '''
        Fit differential equation to data by solving with odeint and
        using fmin to parameters that best fit time/intensity data or
        by using optimize.curve_fit. Different variants have use
        different free parameters.
        '''

        '''
        Variant 1: fits differential equation to data, free parameters
        kOff, Nss, kPh, assumes infinite cytoplasmic pool
        '''
        if 1 in variants:
            def dy_dt(y, t, kOff, Nss, kPh):
                # Calculates derivative for known y and params
                return kOff*Nss-(kOff+kPh)*y

            def objFunc(params, fitTimes, fitData):
                # Returns distance between solution of diff. equ. with
                # parameters params and the data fitData at times fitTimes
                # Do integration of dy_dt using parameters params
                y = integrate.odeint(dy_dt, params[1], fitTimes,
                                     args=(params[0], params[1], params[2]))
                # Get y-values at the times needed to compare with data
                return sum((transpose(y)-fitData)**2)
            # Set reasonable starting values for optimization
            kOffStart, NssStart, kPhStart = 0.01, 100, 0.01
            # Optimize objFunc to find optimal kOffStart, NssStart, kPhStart
            x = optimize.fmin(objFunc, [kOffStart, NssStart, kPhStart],
                              args=(self.fitTimes, self.partCount),
                              disp=False)
            self.kOffVar1, self.NssVar1, self.kPhVar1 = (x[0], x[1], x[2])
            # Get solution using final parameter set determined by fmin
            self.fitSolVar1 = integrate.odeint(dy_dt, self.NssVar1,
                                               self.fitTimes,
                                               args=(self.kOffVar1,
                                                     self.NssVar1,
                                                     self.kPhVar1))
        '''
        Variant 2: fits solution of DE to data, fitting N(0), N(Inf) and koff,
        therefore being equivalent to variant=1
        '''
        if 2 in variants:
            def exact_solution(times, koff, count0, countInf):
                return ((count0 - countInf) *
                        exp(-koff*count0/countInf*times) + countInf)
            popt, pcov = optimize.curve_fit(exact_solution, self.fitTimes,
                                            self.partCount)
            self.kOffVar2 = popt[0]
            self.fitSolVar2 = [exact_solution(t, popt[0], popt[1], popt[2])
                               for t in self.fitTimes]

        '''
        Variant 3: fits solution of DE to data, assuming Nss=N(0) and
        Nss_bleach=N(end), only one free parameter: koff
        '''
        if 3 in variants:
            def exact_solution(count0, countInf):
                def curried_exact_solution(times, koff):
                    return ((count0 - countInf) *
                            exp(-koff*count0/countInf*times) + countInf)
                return curried_exact_solution
            popt, pcov = optimize.curve_fit(exact_solution(self.partCount[0],
                                                           self.partCount[-1]),
                                            self.fitTimes, self.partCount)
            self.kOffVar3 = popt[0]
            func = exact_solution(self.partCount[0], self.partCount[-1])
            self.fitSolVar3 = [func(t, popt[0]) for t in self.fitTimes]

        '''
        Variant 4: fits solution of DE to data, fitting off rate and N(Inf),
        leaving N(0) fixed at experimental value
        '''
        if 4 in variants:
            def exact_solution(count0):
                def curried_exact_solution(times, koff, countInf):
                    return ((count0 - countInf) *
                            exp(-koff*count0/countInf*times) + countInf)
                return curried_exact_solution
            kOffStart, countInf = 0.005, 50
            popt, pcov = optimize.curve_fit(exact_solution(self.partCount[0]),
                                            self.fitTimes, self.partCount,
                                            [kOffStart, countInf])
            self.kOffVar4 = popt[0]
            func = exact_solution(self.partCount[0])
            self.fitSolVar4 = [func(t, popt[0], popt[1])
                               for t in self.fitTimes]

        '''
        Variant 5 (according to supplement Robin et al. 2014):
        Includes cytoplasmic depletion, fixes N(0). N corresponds to R,
        Y corresponds to cytoplasmic volume
        '''
        if 5 in variants:
            def exact_solution(count0):
                def curried_exact_solution(times, r1, r2, kPh):
                    return (count0 * ((kPh+r2) / (r2-r1) * exp(r1*times) +
                            (kPh+r1) / (r1-r2) * exp(r2*times)))
                return curried_exact_solution
            popt, pcov = optimize.curve_fit(exact_solution(self.partCount[0]),
                                            self.fitTimes, self.partCount,
                                            [-0.006, -0.1, 0.1], maxfev=10000)
            print(popt)
            print(sqrt(diag(pcov)))
            print(popt/sqrt(diag(pcov)))
            self.kPhVar5 = popt[2]
            self.kOnVar5 = (popt[0]*popt[1]) / self.kPhVar5
            self.kOffVar5 = -(popt[0]+popt[1]) - (self.kOnVar5+self.kPhVar5)
            func = exact_solution(self.partCount[0])
            self.fitSolVar5 = [func(t, popt[0], popt[1], popt[2])
                               for t in self.fitTimes]

        '''
        Variant 6: This tries to circumvent the error made by fixing the
        starting condition to the first measurement. This point already has a
        statistical error affiliated with it. Fixing it propagates this
        error through the other parameters/the whole fit. Otherwise
        equivalent to variant 5.
        '''
        if 6 in variants:
            def curried_exact_solution(times, r1, r2, kPh, count0):
                return (count0 * ((kPh+r2) / (r2-r1) * exp(r1*times) +
                        (kPh+r1) / (r1-r2) * exp(r2*times)))
            popt, pcov = optimize.curve_fit(curried_exact_solution,
                                            self.fitTimes, self.partCount,
                                            [-0.1, -0.2, -0.01, 200],
                                            maxfev=10000)
            self.count0Var6 = popt[3]
            self.kPhVar6 = popt[2]
            self.kOnVar6 = (popt[0] * popt[1]) / self.kPhVar6
            self.kOffVar6 = -(popt[0] + popt[1]) - (self.kOnVar6 +
                                                    self.kPhVar6)
            self.fitSolVar6 = [curried_exact_solution(
                                t, popt[0], popt[1], self.kPhVar6,
                                self.count0Var6) for t in self.fitTimes]

        if self.showFigs:
            for i in variants:
                self.plot_offRateFit(variant=i)

    def synthetic_offRate_data(self, kon, kOff, kPh, endtime, noise):
        '''
        noise in percent of R0 (initial value of particle count)
        '''
        # Derivative
        def dy_dt(y, t):
            R = y[0]
            Y = y[1]
            f0 = kon*Y - (kOff+kPh)*R
            f1 = -kon*Y + kOff*R
            return[f0, f1]
        # Initial conditions and time grid
        R0 = 300
        Y0 = kOff/kon*R0
        y0 = [R0, Y0]
        self.fitTimes = arange(0, endtime, self.timestep)
        # Solve ode system
        soln = integrate.odeint(dy_dt, y0, self.fitTimes)
        self.partCount = soln[:, 0]
        self.cytoCount = soln[:, 0]
        # Add noise to solution
        self.partCount = (self.partCount + noise * R0 *
                          random.rand(self.partCount.size))

    def plot_offRateFit(self, variant):
        font = {'weight': 'bold',
                'size': 'larger'}
        self.set_fig_style()
        fig, ax = subplots()
        fig.suptitle('Variant ' + str(variant), fontsize=20, fontdict=font,
                     bbox=dict(facecolor='green', alpha=0.3))
        if variant == 1:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar1)
        elif variant == 2:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar2)
        elif variant == 3:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar3)
        elif variant == 4:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar4)
        elif variant == 5:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar5)
        elif variant == 6:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar6)
        else:
            print('Variant ' + str(variant) + ' does not exist.')
        ax.set(ylabel='# particles', xlabel='t [s]')
        if self.saveFigs:
            savefig(self.stackPath + '_offRateFit_variant_' + str(variant) +
                    '.pdf', bbox_inches='tight')
        if self.showFigs:
            show()
        close()

    def gData(self):
        # Make sure things are in alphabetical order, see append_output_to_csv
        return {'File': self.stackPath,
                'kOff1': self.kOffVar1, 'kOff2': self.kOffVar2,
                'kOff3': self.kOffVar3, 'kOff4': self.kOffVar4,
                'kOff5': self.kOffVar5, 'kOff6': self.kOffVar6,
                'kOnVar5': self.kOnVar5, 'kOnVar6': self.kOnVar6,
                'kPh1': self.kPhVar1, 'kPh5': self.kPhVar5,
                'kPh6': self.kPhVar6}


class ParameterSampler():

    def __init__(self, kOffRange, kPhRange, kOnRange, noiseRange, name='1'):
        self.dfInput = DataFrame(columns=['kOff', 'kPh', 'kOn', 'noise'])
        self.dfOutput = DataFrame(columns=['kOffVar1', 'kOffVar2', 'kOffVar3',
                                           'kOffVar4', 'kOffVar5', 'kOffVar6',
                                           'kPhVar1', 'kPhVar5', 'kPhVar6',
                                           'kOnVar5', 'kOnVar6'])
        s = OffRateFitter()
        for kOff, kPh, kOn, noise in product(kOffRange, kPhRange,
                                             kOnRange, noiseRange):
            s.synthetic_offRate_data(kOn, kOff, kPh, 2/kOff, noise)
            try:
                s.fit_offRate()
            except (RuntimeError, ValueError):
                s.kOffVar1 = float('nan')
                s.kOffVar2 = float('nan')
                s.kOffVar3 = float('nan')
                s.kOffVar4 = float('nan')
                s.kOffVar5 = float('nan')
                s.kOffVar6 = float('nan')
                s.kPhVar1 = float('nan')
                s.kPhVar5 = float('nan')
                s.kPhVar6 = float('nan')
                s.kOnVar5 = float('nan')
                s.kOnVar6 = float('nan')
            temp = DataFrame([[kOff, kPh, kOn, noise]],
                             columns=['kOff', 'kPh', 'kOn', 'noise'])
            self.dfInput = self.dfInput.append(temp)
            temp = DataFrame([[s.kOffVar1, s.kOffVar2, s.kOffVar3,
                              s.kOffVar4, s.kOffVar5, s.kOffVar6,
                              s.kPhVar1, s.kPhVar5, s.kPhVar6,
                              s.kOnVar5, s.kOnVar6]],
                             columns=['kOffVar1', 'kOffVar2', 'kOffVar3',
                                      'kOffVar4', 'kOffVar5', 'kOffVar6',
                                      'kPhVar1', 'kPhVar5', 'kPhVar6',
                                      'kOnVar5', 'kOnVar6'])
            self.dfOutput = self.dfOutput.append(temp)
        self.dfOutput.to_csv('out' + str(name) + '.csv')
        self.dfInput.to_csv('in' + str(name) + '.csv')


'''
Convenience functions
'''


def vel_autocorr(x, delta):
    '''
    Returns autocorrelation for all lag times
    '''
    v = array(x)[delta:]-array(x)[:-delta]
    ac = []
    for i in range(len(v)):
        dotted = dot(v[:-i or None], v[i:])
        ac_i = dotted/(len(v[:-i or None]))
        ac.append(ac_i)
    ac = ac/max(ac)
    return ac
