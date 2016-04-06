
from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

# the following line only works in an IPython notebook
%matplotlib notebook

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
from scipy import optimize

import pims
import trackpy as tp

frames = pims.ImageSequence(
            '/Users/hubats01/Desktop/SPT/Us/NWG0026_crt90_DMSO/16_04_05/fov9/*.tif', 
            as_grey=True)

timestep = 0.033
pixelsize = 0.155
minm = 600
featSize = 3
mem = 7
dist = 5
fig1 = plt.figure()
plt.imshow(frames[1])
f = tp.locate(frames[1], featSize, invert=False, minmass = minm)
tp.annotate(f, frames[1])

f = tp.batch(frames[:], featSize, minmass=minm, invert=False)
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
#np.savetxt('/Users/hubats01/Desktop/Single_Molecule_Analysis/Francois/msd_python.csv', numpy_array, delimiter=',')
def func(t, D, a):
    return 4*D*t**a
time = np.linspace(timestep, timestep*numFrames, numFrames)
DA = np.zeros([numParticles, 2])
for i in range(0, numParticles):
    MSD = numpy_array[0:numFrames, i]
    popt, pcov = optimize.curve_fit(func, time, MSD)
    DA[i, ] = popt
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(DA[:, 1], DA[:, 0], '.')
D = DA[:, 0]
a = DA[:, 1]
np.mean(D[(a>0.9) & (a<1.1)])
