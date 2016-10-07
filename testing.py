
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


# # Tests for particle tracking and off rate fitting
# 
# In order to test changes made to particle tracking and off rate fitting this document can serve as a standard. Sample data for diffusion contains PAR-6 measurements, off rate data is taken from PAR-2. Also compare to Matlab's testing file. Both should give roughly the same outputs.

# In[1]:

import time
import sys
from matplotlib import pyplot as plt
sys.path.append('/Users/hubatsl/Desktop/SPT/Us/SPT/Python')
from MovieTracks import DiffusionFitter, OffRateFitter
get_ipython().magic('matplotlib inline')


# # Extracting diffusion coefficients for indiv. particles.
# First, we test the particle tracking by running on the folder specified in 'fol'. 
# After creating an instance of DiffusionFitter (d), d.analyze() is run to find features and link tracks.
# 

# In[ ]:

fol = '/Users/hubatsl/Desktop/SPT/Us/SPT/sample_data/16_07_20_PAR6_2/fov1/'
d = DiffusionFitter(fol, 300, parallel=True, pixelSize=0.120, timestep=0.033,
                    saveFigs=False, showFigs=False, autoMetaDataExtract=False)
t0 = time.time()
d.analyze()
t = time.time() - t0
print('Test took ' + str(t) + ' seconds, normal time ~23 s.')


# **Plot calibration of feature finding for one frame (1st frame by default).**

# In[ ]:

d.showFigs = True
d.plot_calibration()
if d.features.size == 571230:
    print('Total number of features ' + str(d.features.size) + ', as expected.')
else:
    print('Total number of features ' + str(d.features.size) + ', not as expected 571230.')


# **Plot trajectories that are longer than treshold set by user.**

# In[ ]:

d.plot_trajectories()
if d.trajectories.particle.unique().size == 117:
    print('Total number of features ' + str(d.trajectories.particle.unique().size) + ', as expected.')
else:
    print('Total number of features ' + str(d.trajectories.particle.unique().size) + ', not as expected 117.')


# **Plot mean square displacement over time.**

# In[ ]:

d.plot_msd()


# **Finally, fit $\langle x \rangle = 4Dt^\alpha$ and plot D vs $\alpha$**

# In[ ]:

d.plot_diffusion_vs_alpha()


# In[ ]:

if d.D.mean()==0.12002050139512239:
    print('Mean d is ' + str(d.D.mean()) + ', as expected.')
else:
    print('Mean d is ' + str(d.D.mean()) + ', not as expected 0.12002050139512239.')


# # Get off rate according to Robin et al. 2014

# In[2]:

# fol = '/Users/hubatsl/Desktop/SPT/Us/SPT/sample_data/16_05_09_TH120_PAR2/fov3/'
fol = '/Users/hubatsl/Desktop/SPT/Us/OffRate/16_04_11_KK1248xTH110/fov2'
o = OffRateFitter(fol, 40, parallel=True, pixelSize=0.12, timestep=2,
                    saveFigs=True, showFigs=True, autoMetaDataExtract=False)
s = OffRateFitter(fol, 40, parallel=True, pixelSize=0.12, timestep=2,
                    saveFigs=True, showFigs=True, autoMetaDataExtract=False)
t0 = time.time()
o.analyze()
t = time.time() - t0
print('time elapsed: '+ str(t)+'s, should not be substantially bigger than 2 s.')


# Fit bleaching behavior of embryo to $$\frac{dy}{dt}=k_{off}N_{ss}-(k_{off}+k_{bleach})N$$ to extract bleaching and off-rate.

# In[27]:

# o.showFigs = True
# o.plot_calibration(1)
# o.plot_calibration(-1)
s.synthetic_offRate_data(0.0001, 0.007, 0.014, 1000)
s.fit_offRate()


# **Off Rate and Bleaching rate as well as Off Rate calculated by fixing start and end point.**

# In[28]:

# o.kOffVar1, o.kOffVar2, o.kOffVar3, o.kOffVar4, o.kOffVar5
s.kOffVar1, s.kOffVar2, s.kOffVar3, s.kOffVar4, s.kOffVar5


# In[29]:

s.kPhVar1, s.kPhVar5


# In[ ]:

# o.synthetic_offRate_data(0.001, 0.007, 0.007, 1000)
plt.plot(o.synthetic_t, o.synthetic_partCount)


# In[ ]:


# o.synthetic_partCount.size


# The rates should be approximately 0.00358 and 0.0102.
