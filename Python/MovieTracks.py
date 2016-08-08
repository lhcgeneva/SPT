# Author: Lars Hubatsch, object oriented wrapper for trackpy

from functools import partial
from itertools import repeat
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pims
from scipy import optimize, integrate
import seaborn as sns
import trackpy as tp

'''
TO DO:
Write tests
'''


class ParticleFinder:
    '''
    Wraps trackpy so properties of individual movies can be conveniently
    accessed through class interface. Every movie becomes one instance of the
    class. ParticleFinder is a generic interface, for Off Rate or 
    Diffusion Rate use OffRateFitter or DiffusionFitter
    '''
    def __init__(self, filePath, threshold, timestep, featSize=3, dist=5,
                 memory=7, no_workers=8, parallel=True, pixelSize=0.120,
                 saveFigs=False, showFigs=False):
        self.dist = dist
        self.featSize = featSize
        self.filePath = filePath
        print(self.filePath + '*.tif')
        self.frames = pims.ImageSequence(self.filePath + '*.tif', as_grey=True)
        # self.frames = self.frames[1:300]
        self.memory = memory
        self.no_workers = no_workers
        self.parallel = parallel
        self.pixelSize = pixelSize
        self.saveFigs = saveFigs
        self.showFigs = showFigs
        self.threshold = threshold
        self.timestep = timestep

    def analyze(self):
        '''
        Convenience function to run all of the analysis in one go with
        graphs on or off depending on showFigs
        '''
        if self.showFigs: self.plot_calibration()
        self.find_feats()
        self.link_feats()
        self.analyze_tracks()
        if self.showFigs: self.plot_trajectories()
        if self.showFigs: self.plot_msd()
        if self.showFigs: self.plot_diffusion_vs_alpha()
        self.save_output()

    def find_feats(self):
        '''
        Checks whether parallel execution is on, otherwise normal batch
        processing
        '''
        if self.parallel:
            # Create list of frames to be analysed by separate processes
            self.f_list = []
            # Get size of chunks
            s = math.ceil(len(self.frames) / self.no_workers)
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
            self.result = pd.concat(res)
            pool.close()
            pool.join()
        else:
            self.result = tp.batch(self.frames[:], self.featSize,
                                   minmass=self.threshold, invert=False)

    def func_squared(self, t, D, a):
        return 4 * D * t ** a

    def link_feats(self):
        '''
        Link individual frames to build trajectories, filter out stubs shorter
        than 80. Get Mean Square Displacement (msd)
        '''
        t = tp.link_df(self.result, self.dist, memory=self.memory)
        self.trajectories = tp.filter_stubs(t, 80)
        # Get msd microns per pixel = 100/285., frames per second = 24
        self.im = tp.imsd(self.trajectories, self.pixelSize, 1 / self.timestep)

    def analyze_tracks(self):
        # Convert to numpy, get diffusion coefficients
        numParticles = self.trajectories['particle'].nunique()
        numFrames = 10
        imAsNumpyArray = self.im.as_matrix()
        self.time = np.linspace(self.timestep,
                                self.timestep * numFrames, numFrames)
        DA = np.zeros([numParticles, 2])
        for j in range(0, numParticles):
            MSD = imAsNumpyArray[0:numFrames, j]
            popt, pcov = optimize.curve_fit(self.func_squared, self.time, MSD)
            DA[j, ] = popt
        self.D = DA[:, 0]
        self.a = DA[:, 1]

    def plot_calibration(self):
        sns.set_context("poster")
        sns.set_style("dark")
        # fig = plt.figure()
        calibrationFrame = 1
        plt.imshow(self.frames[calibrationFrame])
        f = tp.locate(self.frames[calibrationFrame], self.featSize,
                      invert=False, minmass=self.threshold)
        fig = tp.annotate(f, self.frames[calibrationFrame])
        if self.saveFigs:
            fig1 = fig.get_figure()
            fig1.savefig(self.filePath + 'Particle_Calibration' + '.pdf', bbox_inches='tight')

    def plot_diffusion_vs_alpha(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.a, self.D, '.')
        if self.saveFigs:
            plt.savefig(self.filePath + 'Particle_D_a.pdf', bbox_inches='tight')
        fig.show()

    def plot_msd(self):
        sns.set_context("poster")
        sns.set_style("dark")
        fig, ax = plt.subplots()
        ax.plot(self.im.index, self.im, 'k-', alpha=0.4)
        ax.set(ylabel='$\Delta$ $r^2$ [$\mu$m$^2$]',
               xlabel='lag time $t$', ylim=[0.001, 10])
        ax.set_xscale('log')
        ax.set_yscale('log')
        fig.show()
        if self.saveFigs:
            plt.savefig(self.filePath + 'Particle_msd.pdf', bbox_inches='tight')

    def plot_trajectories(self):
        fig = tp.plot_traj(self.trajectories, label=True)
        fig = fig.get_figure()
        if self.saveFigs:
            fig.savefig(self.filePath + 'Particle_Trajectories' + '.pdf', bbox_inches='tight')

    def save_output(self):
        columns = ['a', 'D']
        combinedNumpyArray = np.c_[self.a, self.D]
        d = pd.DataFrame(data=combinedNumpyArray, columns=columns)
        d.to_csv(self.filePath + 'Particle_D_a.csv')

class OffRateFitter(ParticleFinder):
    '''
    Extends ParticleFinder to implement Off rate calculation
    '''
    def __init__(self):
        super(OffRateFitter, self).__init__()

    def fit_offRate(self):
        '''
        Fit differential equation to data by solving with ode45 and 
        applying minimum least squares
        '''
        def objFunc(x, fitTimes, fitData):
            pass
            # [t,y] = integrate.ode45(@(t,y) x(1)*x(2) - (x(1)+x(3))*y, [min(fitTimes) max(fitTimes)], x(2));   
        #     y_interp = interp1(t, y, fitTimes)
        #     f = sum((y_interp-fitData).^2)
        #     return f
        # kOffStart = 0.001
        # kPhStart = 0.01
        # NssStart = 1
        # x = fminsearch(@(x) objFunc(x, fitTimes, fitData), [kOffStart, NssStart, kPhStart])
        # [t,y] = ode45(@(t,y) x(1)*x(2) - (x(1)+x(3))*y, [min(fitTimes) max(fitTimes)], x(2)
        # plot(t,y,'-', 'LineWidth', 3)
        '''
        Still to be implemented, above code is taken over from matlab
        '''
class DiffusionFitter(ParticleFinder):
    '''
    Extends ParticleFinder to implement Off rate calculation
    '''
    def __init__(self):
        super(DiffusionFitter, self).__init__()

    def fit_offRate(self):
        pass


class StructJanitor:
    '''
    Reads workspace and figures out newly placed files to run tracking on
    '''
    def __init__(self):
        pass

    def read_project_structure():
        pass

    def run_on_new_files(newFileNames):
        for name in newFileNames:
            m = ParticleFinder(name, 600, 0.033)
            m.find_feats()
            m.link_feats()

    def write_project_file():
        pass
