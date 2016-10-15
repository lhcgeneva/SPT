
# coding: utf-8

# In[1]:

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


# # Test for extracting diffusion coefficients via  particle tracking
# 
# In order to test changes made to particle tracking and off rate fitting this document can serve as a standard. Sample data for diffusion contains PAR-6 measurements. Also compare to Matlab's testing file. Both should give roughly the same outputs.

# In[2]:

import time
import sys
from matplotlib import pyplot as plt
sys.path.append('/Users/hubatsl/Desktop/SPT/Us/SPT/Python')
from MovieTracks import DiffusionFitter, OffRateFitter, ParameterSampler
get_ipython().magic('matplotlib inline')


# First, we test the particle tracking by running on the folder specified in 'fol'. 
# After creating an instance of DiffusionFitter (d), d.analyze() is run to find features and link tracks.

# In[3]:

fol = '/Users/hubatsl/Desktop/SPT/Us/SPT/sample_data/16_07_20_PAR6_2/fov1/'
d = DiffusionFitter(fol, 300, parallel=True, pixelSize=0.120, timestep=0.033,
                    saveFigs=False, showFigs=False, autoMetaDataExtract=False)
t0 = time.time()
d.analyze()
t = time.time() - t0
print('Test took ' + str(t) + ' seconds, normal time ~23 s.')


# **Plot calibration of feature finding for one frame (1st frame by default).**

# In[4]:

d.showFigs = True
d.plot_calibration()
if d.features.size == 571230:
    print('Total number of features ' + str(d.features.size) + ', as expected.')
else:
    print('Total number of features ' + str(d.features.size) + ', not as expected 571230.')


# **Plot trajectories that are longer than treshold set by user.**

# In[5]:

d.plot_trajectories()
if d.trajectories.particle.unique().size == 117:
    print('Total number of features ' + str(d.trajectories.particle.unique().size) + ', as expected.')
else:
    print('Total number of features ' + str(d.trajectories.particle.unique().size) + ', not as expected 117.')


# **Plot mean square displacement over time.**

# In[6]:

d.plot_msd()


# **Finally, fit $\langle x \rangle = 4Dt^\alpha$ and plot D vs $\alpha$**

# In[7]:

d.plot_diffusion_vs_alpha()
d.D_restricted


# In[8]:

if d.D.mean()==0.12002050139512239:
    print('Mean d is ' + str(d.D.mean()) + ', as expected.')
else:
    print('Mean d is ' + str(d.D.mean()) + ', not as expected 0.12002050139512239.')


# In[9]:

import numpy
n, bins, patches = plt.hist(part_count.asobject, range(80, 500, 10))
plt.show()

