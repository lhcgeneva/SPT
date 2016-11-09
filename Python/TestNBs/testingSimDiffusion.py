
# coding: utf-8

# In[44]:

get_ipython().magic('qtconsole')


# In[1]:

import sys
sys.path.append('/Users/hubatsl/Desktop/SPT/Us/SPT/Python/src')

import bisect
from IPython.core.debugger import Tracer
import itertools
import matplotlib.pyplot as plt
from MovieTracks import DiffusionFitter
from multiprocessing import Pool
from numba import jit
import numpy as np
import scipy.stats as stats
from SimDiffusion import ImageSimulator, sum_gaussians
import tifffile
import time
import scipy.stats
get_ipython().magic('matplotlib inline')


# **Identifiying different diffusive species**
# In order to distinguish between differently diffusive species I try to compare between experiments like this: 

# In[2]:

a = np.ones((10, 1))
for i in range(10):
    file = '/Users/hubatsl/Desktop/SPT/Us/SPT/sample_data/SyntheticData/test1'
    im = ImageSimulator(np.array([0.2, 0.4]), np.array([100, 100]), fname=file+'/test',timemax=6)
    im.create_images()
    im.write_images()
    im.write_log()
    d = DiffusionFitter(file, 700, parallel=True, pixelSize=0.124, timestep=0.033,
                        saveFigs=False, showFigs=False, autoMetaDataExtract=False)
    d.analyze()
    a[i] = d.D_restricted


# In[3]:

a.std()


# In[4]:

0.319/0.316


# In[99]:

d = DiffusionFitter(file, 700, parallel=True, pixelSize=0.124, timestep=0.033,
                    saveFigs=False, showFigs=False, autoMetaDataExtract=False)
d.analyze()


# In[5]:

d.showFigs = True
d.plot_calibration()
d.plot_diffusion_vs_alpha()
plt.figure(); plt.hist(d.D,50)
plt.figure(); plt.hist(d.a, 25)
d.D_restricted


# In[9]:

b = np.ones((10, 1))
for i in range(10):
    fol= '/Users/hubatsl/Desktop/SPT/Us/SPT/sample_data/SyntheticData/MatlabSample/'+str(i+1)+'/'
    d = DiffusionFitter(fol, 700, parallel=True, pixelSize=0.124, timestep=0.033,
                        saveFigs=False, showFigs=False, autoMetaDataExtract=False)
    d.analyze()
    b[i] = d.D_restricted


# In[23]:

b.std()
print(b)
print(a)
scipy.stats.ks_2samp(b.flatten(), a.flatten())


# In[22]:

b.flatten()


# In[18]:

n1 = 200  # size of first sample
n2 = 300  # size of first sample
rvs1 = stats.norm.rvs(size=n1, loc=0., scale=1)
rvs2 = stats.norm.rvs(size=n2, loc=0.5, scale=1.5)
stats.ks_2samp(rvs1, rvs2)
rvs1


# In[6]:

d.showFigs = True
d.plot_calibration()
d.plot_diffusion_vs_alpha()
plt.figure(); plt.hist(d.D,50)
plt.figure(); plt.hist(d.a, 25)
d.D_restricted


# In[ ]:

plt.plot(np.logspace(0.001, 6, 20, base=4)/10000)

