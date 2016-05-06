from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import os
from multiprocessing import Pool, Process, Queue
import matplotlib as mpl
import matplotlib.pyplot as plt
import trackpy as tp
# the following line only works in an IPython notebook
%matplotlib notebook
%matplotlib inline
# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
from scipy import optimize
import pims
import math
def func_squared(t, D, a):
    return 4*D*t**a
no_movs = 3
no_workers = 8
root_dir = '/Users/hubats01/Desktop/SPT/Us/Diffusion/PAR3/16_04_14_PAR3_WT_NotAlive/'
parallel = True
for minm in [600]:
    for i in range(1, no_movs+1):
        frames = pims.ImageSequence(root_dir+'fov'+str(i)+'/*.tif', as_grey=True)
        timestep = 0.033
        pixelsize = 0.120
        featSize = 3
        mem = 7
        dist = 5
        fig1 = plt.figure()
        plt.imshow(frames[0])
        f = tp.locate(frames[0], featSize, invert=False, minmass = minm)
        fig = tp.annotate(f, frames[0])
        fig1 = fig.get_figure()
        fig1.savefig(root_dir+'Thresh_'+str(i)+'_'+str(minm)+'.pdf', bbox_inches='tight')
        if parallel:
            #Create list of frames to be analysed by separate processes
            f_list = []
            def single_Arg_Batch(fs):
                return tp.batch(fs, 3, minm, invert=False)
            s = math.floor(len(frames)/no_workers) #Get size of chunks
            for ii in range(0, no_workers-1):
                f_list.append(frames[s*ii:s*(ii+1)])#Issue with index, check output!
            #Last worker gets slightly more frames
            f_list.append(frames[s*(no_workers-1):-1])
            pool = Pool(processes = no_workers)
            ret = pool.map(single_Arg_Batch, f_list)
            result = pd.concat(ret)
            pool.close()
            pool.join()
        else:
            result = tp.batch(frames[:], featSize, minmass=minm, invert=False)
        t = tp.link_df(result, dist, memory=mem)
        t1 = tp.filter_stubs(t, 80)
        fig = tp.plot_traj(t1)
        fig1 = fig.get_figure()
        fig1.savefig(root_dir+'Traj_'+str(i)+'_'+str(minm)+'.pdf', bbox_inches='tight')
        tm = t1
        im = tp.imsd(tm, pixelsize, 1/timestep)  # microns per pixel = 100/285., frames per second = 24
        if False:
            fig, ax = plt.subplots()
            ax.plot(im.index, im, 'k-', alpha=0.4)  # black lines, semitransparent
            ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
                   xlabel='lag time $t$', ylim=[0.001,10])
            ax.set_xscale('log')
            ax.set_yscale('log')
        numParticles = t1['particle'].nunique()
        numFrames = 10
        numpy_array = im.as_matrix()
        time = np.linspace(timestep, timestep*numFrames, numFrames)
        DA = np.zeros([numParticles, 2])
        for j in range(0, numParticles):
            MSD = numpy_array[0:numFrames, j]
            popt, pcov = optimize.curve_fit(func_squared, time, MSD)
            DA[j, ] = popt
        if True:
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            ax2.plot(DA[:, 1], DA[:, 0], '.')
            plt.savefig(root_dir+'DiffAlpha_'+str(i)+'_'+str(minm)+'.png', bbox_inches='tight')
        D = DA[:, 0]
        a = DA[:, 1]
        #np.mean(D[(a>0.9) & (a<1.2)])
        np.savetxt(root_dir+'a'+str(i)+'_'+str(minm)+'.csv', a, delimiter=',')
        np.savetxt(root_dir+'D'+str(i)+'_'+str(minm)+'.csv', D, delimiter=',')
