
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

# In[3]:

from IPython.html.widgets import interact, IntSlider
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys
import tifffile
import time
sys.path.append('/Users/hubatsl/Desktop/SPT/Us/SPT/Python/src')
from MovieTracks import (DiffusionFitter, OffRateFitter, ParameterSampler,
                        ParticleFinder)
get_ipython().magic('matplotlib inline')


# **Run thresholding on movie movieNo, read and write threshold from excel file**

# In[ ]:

exc = pd.read_excel('/Users/hubatsl/Desktop/SPT.xlsx')
df_init = pd.read_excel('/Users/hubatsl/Desktop/SPT_input_params.xlsx')
movieNo = 18
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

# In[ ]:

df_init['IntervalReal']='nan'
df_init['ExposureReal']='nan'
for i, path in enumerate(exc['Folder']):
    p = ParticleFinder(path)
    df_init['IntervalReal'][i]=p.timestep
    df_init['ExposureReal'][i]=p.exposure
df_init.to_excel('/Users/hubatsl/Desktop/SPT_input_params.xlsx')


# In[ ]:

exc = pd.read_excel('/Users/hubatsl/Desktop/SPT.xlsx')
df_init = pd.read_excel('/Users/hubatsl/Desktop/SPT_input_params.xlsx')


# In[ ]:

for i, path in enumerate(exc['Folder']):
    interval = df_init['IntervalReal'][i]
    if 2.64/interval<20: mTL = 20
    else: mTL = 2.64/interval
    p = DiffusionFitter(path, df_init['Threshold'][i],
                        minTrackLength=mTL)
    p.analyze()
    p.save_output()
    p.save_summary_input()


# In[ ]:

# for i in range(19, 26):
i = 18
path = exc['Folder'][i]
interval = df_init['IntervalReal'][i]
if 2.64/interval<20: mTL = 20
else: mTL = 2.64/interval
p = DiffusionFitter(path, df_init['Threshold'][i],
                        minTrackLength=mTL, startFrame=200)
p.plot_calibration()
p.analyze()
p.save_output()
p.save_summary_input()


# **Plotting all different D vs alpha in one plot**

# In[ ]:

df_list = []
for i, folder in enumerate(exc['Folder']):
    df_list.append(pd.read_csv(folder+'Particle_D_a.csv'))

get_ipython().magic('matplotlib qt')

f=plt.figure()
sm100 = [df_list[x] for x in range(len(exc['Folder'])) 
       if df_init['IntervalReal'][x]<0.1 ]
sm100 = pd.concat(sm100)
sm200 = [df_list[x] for x in range(len(exc['Folder'])) 
       if df_init['IntervalReal'][x]<0.2 and df_init['IntervalReal'][x]>0.1]
sm200 = pd.concat(sm200)
sm400 = [df_list[x] for x in range(len(exc['Folder'])) 
       if df_init['IntervalReal'][x]<0.4  and df_init['IntervalReal'][x]>0.2]
sm400 = pd.concat(sm400)
sm600 = [df_list[x] for x in range(len(exc['Folder'])) 
       if df_init['IntervalReal'][x]<0.6  and df_init['IntervalReal'][x]>0.4]
sm600 = pd.concat(sm600)
sm2000= [df_list[x] for x in range(len(exc['Folder'])) 
       if df_init['IntervalReal'][x]>0.6 ]
sm2000 = pd.concat(sm2000)
plt.plot(sm100.a, sm100.D, 'b.')
plt.plot(sm200.a, sm200.D, 'r.')
plt.plot(sm400.a, sm400.D, 'y.')
plt.plot(sm600.a, sm600.D, 'm.')
plt.plot(sm2000.a, sm2000.D, 'k.')
plt.xlabel('alpha')
plt.ylabel('D [mu^2/s]')

import pylab as p

plt.figure()

y,binEdges=np.histogram(sm100.a,bins=20, normed=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
p.plot(bincenters,y, 'b-', label='<100ms')
print(np.mean(y))
print(np.std(y))
y,binEdges=np.histogram(sm200.a,bins=20, normed=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
p.plot(bincenters,y,'r-', label='<200ms')

y,binEdges=np.histogram(sm400.a,bins=20, normed=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
p.plot(bincenters,y,'y-', label='<400ms')

y,binEdges=np.histogram(sm600.a,bins=20, normed=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
p.plot(bincenters,y,'m-', label='<600ms')

y,binEdges=np.histogram(sm2000.a,bins=20, normed=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
p.plot(bincenters,y,'k-', label='>600ms')

p.xlabel('alpha')
p.ylabel('normalized counts')
legend = p.legend(loc='upper right', shadow=True)
p.show()


# In[ ]:

df_list = []
for i, folder in enumerate(exc['Folder']):
    df_list.append(pd.read_csv(folder+'Particle_D_a_secondhalf.csv'))

get_ipython().magic('matplotlib qt')

f=plt.figure()
sm100 = [df_list[x] for x in range(len(exc['Folder'])) 
       if df_init['IntervalReal'][x]<0.1 ]
sm100 = pd.concat(sm100)
sm200 = [df_list[x] for x in range(len(exc['Folder'])) 
       if df_init['IntervalReal'][x]<0.2 and df_init['IntervalReal'][x]>0.1]
sm200 = pd.concat(sm200)
sm400 = [df_list[x] for x in range(len(exc['Folder'])) 
       if df_init['IntervalReal'][x]<0.4  and df_init['IntervalReal'][x]>0.2]
sm400 = pd.concat(sm400)
sm600 = [df_list[x] for x in range(len(exc['Folder'])) 
       if df_init['IntervalReal'][x]<0.6  and df_init['IntervalReal'][x]>0.4]
sm600 = pd.concat(sm600)
sm2000= [df_list[x] for x in range(len(exc['Folder'])) 
       if df_init['IntervalReal'][x]>0.6 ]
sm2000 = pd.concat(sm2000)
plt.plot(sm100.a, sm100.D, 'b.')
plt.plot(sm200.a, sm200.D, 'r.')
plt.plot(sm400.a, sm400.D, 'y.')
plt.plot(sm600.a, sm600.D, 'm.')
plt.plot(sm2000.a, sm2000.D, 'k.')
plt.xlabel('alpha')
plt.ylabel('D [mu^2/s]')

import pylab as p

plt.figure()

y,binEdges=np.histogram(sm100.a,bins=20, normed=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
p.plot(bincenters,y, 'b-', label='<100ms')
print(np.mean(y))
print(np.std(y))
y,binEdges=np.histogram(sm200.a,bins=20, normed=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
p.plot(bincenters,y,'r-', label='<200ms')

y,binEdges=np.histogram(sm400.a,bins=20, normed=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
p.plot(bincenters,y,'y-', label='<400ms')

y,binEdges=np.histogram(sm600.a,bins=20, normed=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
p.plot(bincenters,y,'m-', label='<600ms')

y,binEdges=np.histogram(sm2000.a,bins=20, normed=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1]) 
p.plot(bincenters,y,'k-', label='>600ms')

p.xlabel('alpha')
p.ylabel('normalized counts')
legend = p.legend(loc='upper right', shadow=True)
p.show()


# In[ ]:

t = numpy.arange(0.033, 100, 0.033)
msd_sub = 4*0.15*t**0.8
msd_super = 4*0.15*t**1.2
msd_nor = 4*0.15*t**1
plt.figure()
plt.plot(t, msd_sub)
plt.plot(t, msd_super)
plt.plot(t, msd_nor)
plt.xlabel('t')
plt.ylabel('msd')
ax = plt.gca()
# ax.set_xscale('log')
# ax.set_yscale('log')
plt.show()

