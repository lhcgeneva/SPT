
# coding: utf-8

# In[19]:

from IPython.display import HTML

HTML('''<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }

  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>
<script>
  $(document).ready(function(){
    $('div.prompt').hide();
    $('div.back-to-top').hide();
    $('nav#menubar').hide();
    $('.breadcrumb').hide();
    $('.hidden-print').hide();
  });
</script>

<footer id="attribution" style="float:right; color:#999; background:#fff;">
Created with Jupyter, delivered by Fastly, rendered by Rackspace.
</footer>''')


# In[2]:

from IPython.core.debugger import Tracer
from itertools import repeat
from math import ceil
from matplotlib.pyplot import figure, imshow, savefig, subplots, close, show, ioff
from multiprocessing import Pool
from numpy import arange, c_, histogram, interp, linspace, log10, mean, polyfit, sum, zeros
from pandas import concat, DataFrame, read_csv
from pims import ImageSequence
from pims_nd2 import ND2_Reader
from os import chdir, path, walk, makedirs
from re import split
from scipy import optimize, integrate
from statsmodels.api import OLS, add_constant
from sys import exit
from tifffile import imsave

import seaborn as sns
import shutil
import trackpy as tp


# In[3]:

import time
import sys
import numpy as np
import matplotlib.pyplot as mp
sys.path.append('/Users/hubatsl/Desktop/SPT/Us/SPT/Python')
from MovieTracks import DiffusionFitter, OffRateFitter
get_ipython().magic('matplotlib inline')


# In[4]:

fol = '/Users/hubatsl/Desktop/SPT/Us/Diffusion/PAR6/Survived/16_07_20_PAR6_2/'
d = DiffusionFitter(fol, 300, parallel=True, pixelSize=0.1049, timestep=0.042,
                    saveFigs=True, showFigs=False, autoMetaDataExtract=False)
t = []
t0 = time.time()
d.analyze()
t.append(time.time() - t0)
print(t)


# In[11]:

d.showFigs = True
d.plot_diffusion_vs_alpha()


# In[12]:

fol = '/Users/hubatsl/Desktop/SPT/Us/OffRate/16_05_09_TH120/fov1/'
o = OffRateFitter(fol, 60, parallel=True, pixelSize=0.12, timestep=4,
                    saveFigs=True, showFigs=True, autoMetaDataExtract=False)
t = []
t0 = time.time()
o.analyze()
t.append(time.time() - t0)
print(t)


# In[13]:

o.fit_offRate()


# In[ ]:



