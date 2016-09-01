#Author: Lars Hubatsch, object oriented wrapper for trackpy
from IPython.core.debugger import Tracer
from itertools import repeat
from math import ceil
from matplotlib.pyplot import figure, imshow, savefig, subplots, close, show, ioff
from multiprocessing import Pool
from numpy import arange, c_, histogram, linspace, log10, mean, polyfit, sum, transpose, zeros
from pandas import concat, DataFrame, read_csv
from pims import ImageSequence
from pims_nd2 import ND2_Reader
from os import chdir, path, walk, makedirs
from re import split
from scipy import integrate, optimize
from sys import exit
from tifffile import imsave

import seaborn as sns
import shutil
import trackpy as tp

'''
TO DO:
Write tests
'''


class ParticleFinder(object):
    '''
    Wraps trackpy so properties of individual movies can be conveniently
    accessed through class interface. Every movie becomes one instance of the
    class. ParticleFinder is a generic interface, for off rates or 
    diffusion rates use OffRateFitter or DiffusionFitter
    '''
    def __init__(self, filePath, threshold, autoMetaDataExtract=False,
                 dist=5, featSize=3, memory=7, no_workers=8, parallel=True,
                 pixelSize=0.120, saveFigs=False, showFigs=False, timestep=None):
        self.autoMetaDataExtract = autoMetaDataExtract
        self.dist = dist
        self.featSize = featSize
        self.stackPath = filePath
        self.basePath = split('/', self.stackPath[::-1],1)[1][::-1] + '/'
        self.stackName = self.stackPath.split('/')[-1].split('.')[0]
        self.memory = memory
        self.no_workers = no_workers
        self.parallel = parallel
        self.pixelSize = pixelSize
        self.saveFigs = saveFigs
        self.showFigs = showFigs
        self.threshold = threshold
        self.timestep = timestep
        # Check whether to extract frame intervals automatically
        if autoMetaDataExtract and '.nd2' in self.stackPath:
            frames = ND2_Reader(self.stackPath)
            self.timestep = frames[-1].metadata.get('t_ms', None)/(len(frames)-1)
            frames.close()
        elif (autoMetaDataExtract and '.nd2' not in self.stackPath) or self.timestep==None:
            print('Metadata extraction currently only supported for .nd2 files. '+
                  'Please provide the timestep as optional argument to'+
                  'ParticleFinder.')
            exit()
        # If file format of stack is .nd2 read in stack and write images to folder
        if 'nd2' in self.stackPath: self.write_images()
        # Read in image sequence from newly created file
        self.frames = ImageSequence(self.basePath+self.stackName+'/*.tif', as_grey=True)
        # self.frames=self.frames[:199]

    def analyze(self):
        '''
        Convenience function to run analysis in one go with
        graphs on or off depending on showFigs
        '''
        if self.showFigs: self.plot_calibration()
        self.find_feats()

    def find_feats(self):
        '''
        Checks whether parallel execution is on, otherwise normal batch
        processing
        '''
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
                                             repeat(self.threshold)))
            # Concatenate results and close and join pool
            self.features = concat(res)
            pool.close()
            pool.join()
        else:
            self.features = tp.batch(self.frames[:], self.featSize,
                                   minmass=self.threshold, invert=False)

    def plot_calibration(self, calibrationFrame=1):
        self.set_fig_style()
        imshow(self.frames[calibrationFrame])
        f = tp.locate(self.frames[calibrationFrame], self.featSize,
                      invert=False, minmass=self.threshold)
        fig = tp.annotate(f, self.frames[calibrationFrame])
        if self.saveFigs:
            fig1 = fig.get_figure()
            fig1.savefig(self.stackPath + 'Particle_Calibration' + '.pdf',
                         bbox_inches='tight')
        if self.showFigs: show()
        close()
    
    def read_metadata(self):
        '''
        From text file, doesn't work because ND2_Reader crashes a lot, a least in
        shell
        stack = ND2_Reader(self.stackPath)
        metaDataSplit = stack.metadata_text.split()
        ind = metaDataSplit.index('Exposure:')
        self.Exposure = metaDataSplit[ind+1]
        '''
        pass

    def save_summary_to_database(self):
        data = {'dist': self.dist, 'featSize': self.featSize,
                'memory': self.memory, 'no_workers': self.no_workers,
                'parallel': self.parallel, 'pixelSize': self.pixelSize,
                'timestep': self.threshold, 'timestep': self.timestep,
                'path': self.stackPath}
        if path.isfile(self.basePath + 'database.csv'):
            read_frame = read_csv(self.basePath + 'database.csv', index_col=0)
            frame_to_write = DataFrame(data, index=[read_frame.shape[0]])
            frame_to_write = concat([read_frame, frame_to_write])
        else:
            frame_to_write = DataFrame(data, index=[0])
        frame_to_write.to_csv(self.basePath + 'database.csv')
        print('Summary saved.')

    def write_images(self):
        if not path.exists(self.basePath+self.stackName):
            makedirs(self.basePath+self.stackName)
        else:
            print('Directory '+self.basePath+self.stackName+' already exists.')
            fileDelete_yn = input('Delete files and write images? [y/n] ')
            if fileDelete_yn == 'y':
                shutil.rmtree(self.basePath+self.stackName)
                makedirs(self.basePath+self.stackName)
            else:
                exit()
        dir_path = path.dirname(path.realpath(__file__))
        chdir(self.basePath+self.stackName)        
        frames = ND2_Reader(self.stackPath)
        for i in range(len(frames)):
            imsave(self.stackName+'_'+str(i)+'.tif', frames[i])
        chdir(dir_path)
        
    def set_fig_style(self):
        ioff()
        sns.set_context("poster")
        sns.set_style("dark")

class DiffusionFitter(ParticleFinder):
    '''
    Example call for debugging:
    d = DiffusionFitter('/Users/hubatsl/Desktop/DataSantaBarbara/Aug_09_10_Test_SPT/
    th411_P0_40ms_100p_299gain_1678ang_earlyMaintenance.nd2', 1100, autoMetaDataExtract=True)

    Extends ParticleFinder to implement Off rate calculation
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def analyze(self):
        super().analyze()
        self.link_feats()
        self.analyze_tracks()
        if self.showFigs: self.plot_trajectories()
        if self.showFigs: self.plot_msd()
        if self.showFigs: self.plot_diffusion_vs_alpha()
        self.save_output()

    def analyze_tracks(self):
        # Convert to numpy, get diffusion coefficients
        numParticles = self.trajectories['particle'].nunique()
        numFrames = 10
        imAsNumpyArray = self.im.as_matrix()
        self.time = linspace(self.timestep,
                                self.timestep * numFrames, numFrames)
        DA = zeros([numParticles, 2])
        for j in range(0, numParticles):
            MSD = imAsNumpyArray[0:numFrames, j]
            results = polyfit(log10(self.time), log10(MSD), 1)
            DA[j, ] = [results[0], results[1]]
        self.D = 10**DA[:, 1]/4
        self.a = DA[:, 0]
        self.D_restricted = mean(self.D[(self.a > 0.9) & (self.a < 1.2)])

    def link_feats(self):
        '''
        Link individual frames to build trajectories, filter out stubs shorter
        than 80. Get Mean Square Displacement (msd)
        '''
        t = tp.link_df(self.features, self.dist, memory=self.memory)
        self.trajectories = tp.filter_stubs(t, 80)
        # Get msd microns per pixel = 100/285., frames per second = 24
        self.im = tp.imsd(self.trajectories, self.pixelSize, 1 / self.timestep)

    def plot_diffusion_vs_alpha(self):
        grouped = self.trajectories.groupby('particle')
        weights = [mean(group.mass) for name, group in grouped]
        self.set_fig_style()
        fig = figure()
        ax = fig.add_subplot(111)
        # ax.plot(self.a, self.D, '.')
        ax.scatter(self.a, self.D, c=weights)
        if self.saveFigs:
            savefig(self.stackPath + 'Particle_D_a.pdf', bbox_inches='tight')
        if self.showFigs: show()
        close()

    def plot_trajectories(self):
        self.set_fig_style()
        fig = tp.plot_traj(self.trajectories, label=True)
        fig = fig.get_figure()
        if self.saveFigs:
            fig.savefig(self.stackPath + 'Particle_Trajectories' + '.pdf',
                        bbox_inches='tight')
        close()

    def plot_msd(self):
        self.set_fig_style()
        fig, ax = subplots()
        ax.plot(self.im.index, self.im, 'k-', alpha=0.4)
        ax.set(ylabel='$\Delta$ $r^2$ [$\mu$m$^2$]',
               xlabel='lag time $t$', ylim=[0.001, 10])
        ax.set_xscale('log')
        ax.set_yscale('log')        
        if self.saveFigs:
            savefig(self.stackPath + 'Particle_msd.pdf', bbox_inches='tight')
        if self.showFigs: show()
        close()

    def save_output(self):
        columns = ['a', 'D']
        combinedNumpyArray = c_[self.a, self.D]
        d = DataFrame(data=combinedNumpyArray, columns=columns)
        d.to_csv(self.stackPath + 'Particle_D_a.csv')


class OffRateFitter(ParticleFinder):
    '''
    Extends ParticleFinder to implement Off rate calculation
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def analyze(self):
        super().analyze()
        self.partCount, _ = histogram(self.features.frame,
                                      bins=self.features.frame.max()+1)
        self.fitTimes = arange(0, len(self.partCount)*self.timestep,
                               self.timestep)

    def fit_offRate(self):
        '''
        Fit differential equation to data by solving with odeint and 
        using fmin to parameters that best fit time/intensity data
        '''
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
            # Tracer()()
            return sum((transpose(y)-fitData)**2)
        # Set reasonable starting values for optimization
        kOffStart, NssStart, kPhStart = 0.01, 100, 0.01
        # Optimize objFunc to find optimal kOffStart, NssStart, kPhStart
        x = optimize.fmin(objFunc, [kOffStart, NssStart, kPhStart],
                          args=(self.fitTimes, self.partCount))
        self.kOff, self.Nss, self.kPh = (x[0], x[1], x[2])
        # Get solution using final parameter set determined by fmin
        self.fitSol = integrate.odeint(dy_dt, self.Nss, self.fitTimes,
                                       args=(self.kOff, self.Nss, self.kPh))
        # Do plot if needed
        if self.showFigs: self.plot_offRateFit()

    def plot_offRateFit(self):
        self.set_fig_style()
        fig, ax = subplots()
        ax.plot(self.fitTimes, self.partCount, self.fitTimes, self.fitSol)
        ax.set(ylabel='# particles', xlabel='t [s]')   
        if self.saveFigs:
            savefig(self.stackPath + '_offRateFit.pdf', bbox_inches='tight')
        if self.showFigs: show()
        close()

class StructJanitor(object):
    '''
    Reads workspace and figures out newly placed files to run tracking on

    Idea: Make user interact for every single movie to adjust threshold.
    '''
    def __init__(self, basePath):
        dir_path = path.dirname(path.realpath(__file__))
        self.basePath = basePath
        chdir(self.basePath)
        self.dir_list = next(walk('.'))[1]
        self.database = read_csv(self.basePath+'database.csv')
        # bo = [self.database.path.str.contains(x) for x in self.dir_list]
        for directory in self.dir_list:
            bo = self.database.path.str.contains(directory)
            print(bo)
        chdir(dir_path)
    def run_on_new_files(newFileNames):
        # for name in newFileNames:
        #     m = ParticleFinder(name, 600, 0.033)
        #     m.find_feats()
        #     m.link_feats()
        pass

    def write_project_file():
        pass
