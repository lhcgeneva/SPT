# Author: Lars Hubatsch, object oriented wrapper for trackpy
from IPython.core.debugger import Tracer
from itertools import repeat, product
from math import ceil
from matplotlib.pyplot import (close, figure, imshow, ioff, savefig, scatter,
                               show, subplots)
from multiprocessing import Pool
from numpy import (arange, c_, diff, exp, histogram, linspace, log10, mean,
                   polyfit, random, sum, transpose, zeros)
from pandas import concat, DataFrame
from pims import ImageSequence
from pims_nd2 import ND2_Reader
from os import chdir, makedirs, path, stat
from re import findall, split
from scipy import integrate, optimize
from sys import exit
from tifffile import imsave, TiffFile

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
    '''

    def __init__(self, filePath=None, threshold=40,
                 autoMetaDataExtract=True, dist=5, featSize=5, maxsize=None,
                 memory=7, minTrackLength=80, no_workers=8, parallel=True,
                 pixelSize=0.120, saveFigs=False, showFigs=False,
                 startFrame=0, timestep=None):
        self.no_workers = no_workers
        self.parallel = parallel
        self.saveFigs = saveFigs
        self.showFigs = showFigs
        self.timestep = timestep
        if filePath is not None:
            self.autoMetaDataExtract = autoMetaDataExtract
            self.dist = dist
            self.featSize = featSize
            self.maxsize = maxsize
            self.minTrackLength = minTrackLength
            self.startFrame = startFrame
            self.stackPath = filePath
            self.basePath = split('/', self.stackPath[::-1], 1)[1][::-1] + '/'
            self.stackName = self.stackPath.split('/')[-1].split('.')[0]
            self.memory = memory
            self.pixelSize = pixelSize
            self.threshold = threshold
            # Check whether to extract frame intervals automatically
            if autoMetaDataExtract and '.nd2' in self.stackPath:
                frames = ND2_Reader(self.stackPath)
                self.timestep = (frames[-1].metadata.get('t_ms', None) /
                                 (len(frames)-1))
                frames.close()
            elif autoMetaDataExtract and '.stk' in self.stackPath:
                tif = TiffFile(self.stackPath)
                string = tif.pages[0].image_description[0:1000].decode("utf-8")
                idx1 = string.find('Exp')
                idx2 = string.find('ms')
                exp_string = string[idx1:idx2+4]
                self.exposure = float(findall('\d+', exp_string)[0])/1000
                time_created = tif.pages[0].uic2tag.time_created/1000  # in sec
                self.timestep = mean(diff(time_created))
                if timestep is not None:
                    print('Input time step ignored.')
            elif ((autoMetaDataExtract and '.nd2' not in self.stackPath) or
                    self.timestep is None):
                print(''''Metadata extraction currently only supported for
                           .nd2 and .stk files. Please provide the timestep as
                           optional argument to ParticleFinder.''')
                exit()
            # If file format of stack is .nd2 read and write stack
            if 'nd2' or '.stk' or '.tif' in self.stackPath:
                self.write_images()
            # Read in image sequence from newly created file
            self.frames = ImageSequence(self.basePath+self.stackName+'/*.tif',
                                        as_grey=True)
            self.frames = self.frames[self.startFrame:]

    def analyze(self):
        '''
        Convenience function to run analysis in one go with
        graphs on or off depending on showFigs
        '''
        if self.showFigs:
            self.plot_calibration()
        self.find_feats()

    def append_output_to_csv(self, csv_path, data):
        cols = [a for a in data.keys()]
        df = DataFrame(data, index=[0])
        with open(csv_path, 'a') as f:
            # Check whether file empty, if not omit header
            # The columns need to be in alphabetical order, because of pandas
            # bug, should be fixed in next pandas release.
            if stat(csv_path).st_size == 0:
                # Make sure to keep alphabetical order!!!
                df.to_csv(f, header=True, cols=cols)
            else:
                df.to_csv(f, header=False, cols=cols)

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
                                             repeat(self.threshold),
                                             repeat(self.maxsize)))
            # Concatenate results and close and join pool
            self.features = concat(res)
            pool.close()
            pool.join()
        else:
            self.features = tp.batch(self.frames[:], self.featSize,
                                     minmass=self.threshold,
                                     maxsize=self.maxsize,
                                     invert=False)

    def plot_calibration(self, calibrationFrame=0):
        self.set_fig_style()
        imshow(self.frames[calibrationFrame])
        f = tp.locate(self.frames[calibrationFrame], self.featSize,
                      invert=False, minmass=self.threshold,
                      maxsize=self.maxsize)
        fig = tp.annotate(f, self.frames[calibrationFrame])
        if self.saveFigs:
            fig1 = fig.get_figure()
            fig1.savefig(self.stackPath + 'Particle_Calibration' + '.png',
                         bbox_inches='tight')
        if self.showFigs:
            show()
        close()

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
        sns.set_style("dark", {"axes.facecolor": 'c'})

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

    Extends ParticleFinder to implement Off rate calculation
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def analyze(self):
        super().analyze()
        self.link_feats()
        self.analyze_tracks()
        if self.showFigs:
            self.plot_trajectories()
            self.plot_msd()
            self.plot_diffusion_vs_alpha()
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
        than minTrackLength. Get Mean Square Displacement (msd)
        '''
        t = tp.link_df(self.features, self.dist, memory=self.memory)
        self.trajectories = tp.filter_stubs(t, self.minTrackLength)
        # Get msd microns per pixel = 100/285., frames per second = 24
        self.im = tp.imsd(self.trajectories, self.pixelSize, 1 / self.timestep)

    def plot_diffusion_vs_alpha(self, xlim=None, ylim=None):
        grouped = self.trajectories.groupby('particle')
        weights = [mean(group.mass) for name, group in grouped]
        weights1 = self.trajectories.particle.value_counts(sort=False).tolist()
        weights = [x if x <= 100 else 100 for x in weights1]
        self.set_fig_style()
        fig = figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.a, self.D, c=weights, edgecolors='none', alpha=0.5,
                   s=50)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if self.saveFigs:
            savefig(self.stackPath + 'Particle_D_a.pdf', bbox_inches='tight')
        if self.showFigs:
            show()
        close()

    def plot_diffusion_vs_alpha_verbose(self):
        df = DataFrame({'D': self.D, 'a': self.a})
        sns.set(style="white")
        g = sns.PairGrid(df, diag_sharey=False, size=8)
        g.map_lower(sns.kdeplot, cmap="Blues_d")
        g.map_upper(scatter)
        g.map_diag(sns.kdeplot, lw=3)
        show()

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
        fig.suptitle('MSD vs lag time', fontsize=20)
        ax.plot(self.im.index, self.im, 'k-', alpha=0.4)
        ax.set(ylabel='$\Delta$ $r^2$ [$\mu$m$^2$]',
               xlabel='lag time $t$', ylim=[0.001, 10])
        ax.set_xscale('log')
        ax.set_yscale('log')
        if self.saveFigs:
            savefig(self.stackPath + 'Particle_msd.pdf', bbox_inches='tight')
        if self.showFigs:
            show()
        close()

    def save_output(self):
        super().save_summary_input()
        columns = ['a', 'D']
        combinedNumpyArray = c_[self.a, self.D]
        d = DataFrame(data=combinedNumpyArray, columns=columns)
        d.to_csv(self.stackPath + 'D_a_'+self.stackName+'.csv')

    def gData(self):
        return {'Alpha_mean': self.a.mean(), 'D_mean': self.D.mean(),
                'D_restr': self.D_restricted, 'File': self.stackPath}


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
            popt, pcov = optimize.curve_fit(exact_solution(self.partCount[0]),
                                            self.fitTimes, self.partCount)
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
                                            [-0.1, -0.2, -0.3], maxfev=10000)
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
                                            [-0.1, -0.2, -0.3, 200],
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
            savefig(self.stackPath + '_offRateFit.pdf', bbox_inches='tight')
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
