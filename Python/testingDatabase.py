
# coding: utf-8

# In[ ]:

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


# # Test for databank system

# In[60]:

from matplotlib import pyplot as plt
import sys
import tifffile
import time
sys.path.append('/Users/hubatsl/Desktop/SPT/Us/SPT/Python')
from MovieTracks import (DiffusionFitter, OffRateFitter, ParameterSampler,
                        ParticleFinder)
get_ipython().magic('matplotlib inline')


# **Run thresholding on movie movieNo, read and write threshold from excel file**

# In[57]:

from IPython.html.widgets import interact, IntSlider
import pandas

exc = pandas.read_excel('/Users/hubatsl/Desktop/SPT.xlsx')
df_init = pandas.read_excel('/Users/hubatsl/Desktop/SPT_input_params.xlsx')
movieNo = 2
print(df_init['Threshold'][movieNo])
thresh_widget = IntSlider(min=100, max=5000, step=50,
                          value=df_init['Threshold'][movieNo])
calibrationFrame_widget = IntSlider(min=1, max=500, step=1)

    
def adj_thresh(thresh, calibFrame):
    print(exc['Folder'][movieNo])
    o = OffRateFitter(exc['Folder'][movieNo], thresh, pixelSize=0.120)
    o.plot_calibration(calibrationFrame=calibFrame)
    df_init['Threshold'][movieNo] = thresh
    df_init.to_excel('/Users/hubatsl/Desktop/SPT_input_params.xlsx')
    
interact(adj_thresh, thresh=thresh_widget, 
         calibFrame=calibrationFrame_widget,
         continuous_update=False)


# **Read time stamps for each movie, write to excel file**

# In[80]:

df_init['IntervalReal']='nan'
df_init['ExposureReal']='nan'
for i, path in enumerate(exc['Folder']):
    p = ParticleFinder(path)
    df_init['IntervalReal'][i]=p.timestep
    df_init['ExposureReal'][i]=p.exposure
df_init


# In[74]:




# In[ ]:



