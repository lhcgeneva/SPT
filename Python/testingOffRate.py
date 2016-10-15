
# coding: utf-8

# In[15]:

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


# # Tests for off rate fitting
# 
# In order to test changes made to off rate fitting this document can serve as a standard. Sample off rate data is taken from PAR-2. Also compare to Matlab's testing file. Both should give roughly the same outputs.

# In[22]:

import numpy
import sys
import time
from matplotlib import pyplot as plt
sys.path.append('/Users/hubatsl/Desktop/SPT/Us/SPT/Python')
from MovieTracks import OffRateFitter
get_ipython().magic('matplotlib inline')


# # Get off rate according to Robin et al. 2014

# In[23]:

# fol = '/Users/hubatsl/Desktop/SPT/Us/SPT/sample_data/16_05_09_TH120_PAR2/fov3/'
fol = '/Users/hubatsl/Desktop/SPT/Us/OffRate/16_04_11_KK1248xTH110/fov2'
# fol = None
o = OffRateFitter(filePath=fol, threshold=40, parallel=True, pixelSize=0.12, timestep=2,
                    saveFigs=True, showFigs=True, autoMetaDataExtract=False)
t0 = time.time()
o.analyze()
t = time.time() - t0
print('time elapsed: '+ str(t)+'s, should not be substantially bigger than 2 s.')


# Fit bleaching behavior of embryo to $$\frac{dy}{dt}=k_{off}N_{ss}-(k_{off}+k_{bleach})N$$ to extract bleaching and off-rate.

# In[25]:

o.showFigs = True
o.plot_calibration(1)
o.plot_calibration(-1)
o.fit_offRate([1, 2, 3, 4, 5, 6])


# **Off Rate and Bleaching rate as well as Off Rate calculated by fixing start and end point.**

# In[26]:

o.kOffVar1, o.kOffVar2, o.kOffVar3, o.kOffVar4, o.kOffVar5, o.kOffVar6


# In[29]:

s = OffRateFitter(None, 40)
s.synthetic_offRate_data(0.0001, 0.007, 0.014, 300, 0.1)
s.fit_offRate([1])


# In[28]:

import numpy
from MovieTracks import ParameterSampler
#Get noise value
noise = numpy.array([0.1])
#For each noise value run the following param combinations
offRate = numpy.arange(0.001, 0.02, 0.001)
kOn = numpy.arange(0, 500, 100)
kPh = numpy.arange(0.005, 0.15, 0.005)
#Do the sampling
ParameterSampler(offRate, kPh, kOn, noise, '1')


# In[ ]:



