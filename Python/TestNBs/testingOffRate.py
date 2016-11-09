
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


# # Tests for off rate fitting
# 
# In order to test changes made to off rate fitting this document can serve as a standard. Sample off rate data is taken from PAR-2. Also compare to Matlab's testing file. Both should give roughly the same outputs.

# In[2]:

import numpy
import sys
import time
from matplotlib import pyplot as plt
sys.path.append('/Users/hubatsl/Desktop/SPT/Us/SPT/Python/src')
from MovieTracks import OffRateFitter
get_ipython().magic('matplotlib inline')


# # Get off rate according to Robin et al. 2014

# In[3]:

fol = '/Users/hubatsl/Desktop/SPT/Us/OffRate/16_04_11_KK1248xTH110/fov2'
o = OffRateFitter(filePath=fol, threshold=40, parallel=True, pixelSize=0.12, timestep=2,
                    saveFigs=True, showFigs=True, autoMetaDataExtract=False)
t0 = time.time()
o.analyze()
t = time.time() - t0
print('time elapsed: '+ str(t)+'s, should not be substantially bigger than 2 s.')


# Fit bleaching behavior of embryo to $$\frac{dy}{dt}=k_{off}N_{ss}-(k_{off}+k_{bleach})N$$ to extract bleaching and off-rate.

# In[4]:

o.showFigs = True
o.plot_calibration(1)
o.plot_calibration(-1)
o.fit_offRate([1, 2, 3, 4, 5, 6])


# **Off Rate and Bleaching rate as well as Off Rate calculated by fixing start and end point.**

# In[5]:

if ((o.kOffVar1,
     o.kOffVar2,
     o.kOffVar3,
     o.kOffVar4,
     o.kOffVar5,
     o.kOffVar6)==
(0.0046257653293790453,
 0.0046257549944317142,
 0.0032856320149524279,
 0.0046973911755056391,
 0.0072796555394321547,
 0.007430668016203925)):
    print('off rates as expected.')
else: print('off rates not as expected.')


# In[6]:

s = OffRateFitter(None, 40)
s.synthetic_offRate_data(0.0001, 0.007, 0.014, 300, 0.1)
s.showFigs=True
s.fit_offRate([1])


# In[7]:

s = OffRateFitter(None, 40)
s.synthetic_offRate_data(600, 0.0005, 0.0005, 2/0.0005, 1)
s.showFigs=True
s.fit_offRate([1, 6])
s.kPhVar1/0.0005


# ** Test for automatic meta data extraction to get precise time intervals **

# In[8]:

fol = '/Users/hubatsl/Desktop/SPT/Us/SPT/sample_data/16_04_11/100p_1s_100ms.stk'
o = OffRateFitter(filePath=fol, threshold=2000, parallel=True, pixelSize=0.12, timestep=2,
                    saveFigs=True, showFigs=True, autoMetaDataExtract=True)
t0 = time.time()
o.analyze()
t = time.time() - t0
print('time elapsed: '+ str(t)+'s, should not be substantially bigger than 2 s.')
o.plot_calibration()


# In[9]:

o.fitTimes


# In[10]:

o.showFigs = True
o.plot_calibration(1)
# o.plot_calibration(-1)
o.fit_offRate([1, 2, 3, 4, 5, 6])


# In[ ]:



