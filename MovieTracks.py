# Author: Lars Hubatsch, object oriented wrapper for trackpy

# for compatibility with Python 2 and 3
from __future__ import division, unicode_literals, print_function

import math
import matplotlib.pyplot as plt
# from multiprocessing import Pool
from pathos.multiprocessing import Pool as Pool
import numpy as np
import pandas as pd
import pims
from scipy import optimize
import seaborn as sns
import trackpy as tp


class MovieTracks:
    '''
    Wraps trackpy so properties of individual movies can be conveniently
    accessed through class interface. Every movie becomes one instance of the
    class
    '''

    def __init__(self, filePath, threshold, timestep, featSize=3, dist=5,
                 memory=7, no_workers=8, parallel=True, pixelSize=0.120,
                 showFigs=False):
        self.dist = dist
        self.featSize = featSize
        self.filePath = filePath
        self.frames = pims.ImageSequence(self.filePath, as_grey=True)
        self.memory = memory
        self.no_workers = no_workers
        self.parallel = parallel
        self.pixelSize = pixelSize
        self.showFigs = showFigs
        self.threshold = threshold
        self.timestep = timestep

    def track(self):
        self.frames = pims.ImageSequence(self.filePath, as_grey=True)
        if self.showFigs:
            self.plot_calibration()

        # Parallel execution of feature finding procedure
        if self.parallel:
            # Create list of frames to be analysed by separate processes
            self.f_list = []
            # Get size of chunks
            s = math.floor(len(self.frames) / self.no_workers)
            for ii in range(0, self.no_workers - 1):
                # Issue with index, check output!
                self.f_list.append(self.frames[s * ii:s * (ii + 1)])
            # Last worker gets slightly more frames
            self.f_list.append(self.frames[s * (self.no_workers - 1): - 1])
            result = parallelFeatureFinding(self.f_list, self.no_workers,
                                            self.threshold)
        else:
            result = tp.batch(self.frames[:], self.featSize,
                              minmass=self.threshold, invert=False)

        # Link individual frames to build trajectories,
        # filter out stubs shorter than 80
        t = tp.link_df(result, self.dist, memory=self.memory)
        self.t1 = tp.filter_stubs(t, 80)

        # Plot trajectories, save
        import ipdb; ipdb.set_trace()  # breakpoint 0dc0b458 //
        if self.showFigs:
            self.plot_trajectories(self.t1)
        tm = self.t1

        # Get/plot msd microns per pixel = 100/285., frames per second = 24
        self.im = tp.imsd(tm, self.pixelSize, 1 / self.timestep)
        if self.showFigs:
            self.plot_msd()

        # Convert to numpy, get diffusion coefficients
        numParticles = self.t1['particle'].nunique()
        numFrames = 10
        imAsNumpyArray = self.im.as_matrix()
        self.time = np.linspace(self.timestep,
                                self.timestep * numFrames, numFrames)
        DA = np.zeros([numParticles, 2])
        for j in range(0, numParticles):
            MSD = imAsNumpyArray[0:numFrames, j]
            popt, pcov = optimize.curve_fit(self.func_squared, self.time, MSD)
            DA[j, ] = popt
        if self.showFigs:
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            ax2.plot(DA[:, 1], DA[:, 0], '.')
            plt.savefig(self.filePath + 'DiffAlpha.png', bbox_inches='tight')
        self.D = DA[:, 0]
        self.a = DA[:, 1]

    def func_squared(self, t, D, a):
        return 4 * D * t ** a

    def plot_calibration(self):
        sns.set_context("poster")
        sns.set_style("dark")
        fig1 = plt.figure()
        calibrationFrame = 1
        plt.imshow(self.frames[calibrationFrame])
        f = tp.locate(self.frames[calibrationFrame], self.featSize,
                      invert=False, minmass=self.threshold)
        fig = tp.annotate(f, self.frames[calibrationFrame])
        fig1 = fig.get_figure()
        fig1.savefig(self.filePath + 'Thresh_' + '.pdf', bbox_inches='tight')

    def plot_msd(self):
        sns.set_context("poster")
        sns.set_style("dark")
        fig, ax = plt.subplots()
        ax.plot(self.im.index, self.im, 'k-', alpha=0.4)
        ax.set(ylabel='$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
               xlabel='lag time $t$', ylim=[0.001, 10])
        ax.set_xscale('log')
        ax.set_yscale('log')
        fig.show()

    def plot_trajectories(self, trajectories):
        fig = tp.plot_traj(trajectories, label=True)
        fig = fig.get_figure()
        # ax = fig.gca()
        fig.savefig(self.filePath + 'Traj_' + '.pdf', bbox_inches='tight')

    def save_output(self):
        np.savetxt(self.filePath + 'a.csv', self.a, delimiter=',')
        np.savetxt(self.filePath + 'D.csv', self.D, delimiter=',')

    def single_Arg_Batch(self):
        return tp.batch(self.f_list, 3, self.threshold, invert=False)


def parallelFeatureFinding(f_list, no_workers, threshold):
    def single_Arg_Batch(fs):
        return tp.batch(fs, 3, threshold, invert=False)
    pool = Pool(processes=no_workers)
    ret = pool.map(single_Arg_Batch, f_list)
    result = pd.concat(ret)
    pool.close()
    pool.join()
    return result
