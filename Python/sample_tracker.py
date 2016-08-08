
from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import matplotlib as mpl
import matplotlib.pyplot as plt
import trackpy as tp
mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')
import numpy as np
from scipy import optimize

import pims
no_movs = 1
root_dir = '/Users/hubatsl/Desktop/SPT/Us/Diffusion/PAR6/'
for i in range(1, no_movs+1):
    frames = pims.ImageSequence('/Users/hubatsl/Desktop/SPT/sample_data/bulk_water/*.png', as_grey=True)
    timestep = 0.033
    pixelsize = 0.120
    minm = 600
    featSize = 3
    mem = 7
    dist = 5
    fig1 = plt.figure()
    plt.imshow(frames[1])
    f = tp.locate(frames[0], 11, invert=True)
    tp.annotate(f, frames[1])

    f = tp.batch(frames[:300], 11, minmass=200, invert=True);
    t = tp.link_df(f, dist, memory=mem)
    t1 = tp.filter_stubs(t, 80)
    # Compare the number of particles in the unfiltered and filtered data.
    print('Before:', t['particle'].nunique())
    print('After:', t1['particle'].nunique())
    plt.figure()
    tp.plot_traj(t1)

    tm = t1
    im = tp.imsd(tm, pixelsize, 1/timestep)  # microns per pixel = 100/285., frames per second = 24
    fig, ax = plt.subplots()
    ax.plot(im.index, im, 'k-', alpha=0.4)  # black lines, semitransparent
    ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
           xlabel='lag time $t$', ylim=[0.001,10])
    ax.set_xscale('log')
    ax.set_yscale('log')

    numParticles = t1['particle'].nunique()
    numFrames = 10
    numpy_array = im.as_matrix()
    def func(t, D, a):
        return 4*D*t**a
    time = np.linspace(timestep, timestep*numFrames, numFrames)
    DA = np.zeros([numParticles, 2])
    for j in range(0, numParticles):
        MSD = numpy_array[0:numFrames, j]
        popt, pcov = optimize.curve_fit(func, time, MSD)
        DA[j, ] = popt
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(DA[:, 1], DA[:, 0], '.')
    D = DA[:, 0]
    a = DA[:, 1]
    np.mean(D[(a>0.9) & (a<1.2)])
    np.savetxt(root_dir+'a'+str(i)+'.csv', a, delimiter=',')
    np.savetxt(root_dir+'D'+str(i)+'.csv', D, delimiter=',')
