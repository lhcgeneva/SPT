
# coding: utf-8

# In[14]:

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


# # Tests for Parameter Sampling
# 
# In order to test changes made to off rate fitting this document can serve as a standard. Here, off-rate data is simulated for various parameter combinations, fits are performed and the resulting parameter space sweep is analysed.

# In[1]:

import numpy
import sys
import time
from matplotlib import pyplot as plt
sys.path.append('/Users/hubatsl/Desktop/SPT/Us/SPT/Python')
from MovieTracks import ParameterSampler
get_ipython().magic('matplotlib inline')


# In[2]:

import numpy
from MovieTracks import ParameterSampler
#Get noise value
noise = numpy.array([0.1])
#For each noise value run the following param combinations
offRate = numpy.arange(0.01, 0.02, 0.01)
kOn = numpy.arange(100, 200, 100)
kPh = numpy.arange(0.01, 0.02, 0.01)
#Do the sampling
p = ParameterSampler(offRate, kPh, kOn, noise, '1')


# In[3]:

p.dfInput


# In[ ]:



