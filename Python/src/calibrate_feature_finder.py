from ipywidgets import interact
from pandas import read_excel
from numpy import isnan
from MovieTracks import ParticleFinder

# featS and movieNo have to be set interactively in jupyter


def adj_thresh(calibFrame, thresh, maxSize, startFrame):
    '''
    adj_thresh is called by interact every time button is pressed
    '''
    calibFrame = int(calibFrame)
    thresh = int(thresh)
    startFrame = int(startFrame)
    maxSize = int(maxSize)
    if maxSize is not 0:
        maxSize = int(maxSize)
        o = ParticleFinder(xls_db['Folder'][movieNo], thresh,
                           pixelSize=0.120, maxsize=maxSize/10000,
                           startFrame=startFrame, featSize=featS)
    else:
        o = ParticleFinder(xls_db['Folder'][movieNo], thresh, pixelSize=0.120,
                           maxsize=None, startFrame=startFrame, featSize=featS)
    print(xls_db['Folder'][movieNo])
    o.plot_calibration(calibrationFrame=calibFrame)
    xls_db.loc[movieNo, 'StartFrame'] = startFrame
    xls_db.loc[movieNo, 'Threshold'] = thresh
    xls_db.loc[movieNo, 'MaxSize'] = maxSize
    xls_db.loc[movieNo, 'FeatSize'] = featS
    xls_db.to_excel(xls_file)

# Read in initial values from excel file if defined, otherwise get default
xls_db = read_excel(xls_file)
if isnan(xls_db['Threshold'][movieNo]):
    xls_db.loc[movieNo, 'Threshold'] = 3000
thresh = int(xls_db['Threshold'][movieNo])
if isnan(xls_db['StartFrame'][movieNo]):
    xls_db.loc[movieNo, 'StartFrame'] = 0
startFrame = int(xls_db.loc[movieNo, 'StartFrame'])
if isnan(xls_db['MaxSize'][movieNo]):
    xls_db.loc[movieNo, 'MaxSize'] = 0
maxSize = int(xls_db.loc[movieNo, 'MaxSize'])

# interact/IPython widget
interact(adj_thresh, thresh=str(thresh), calibFrame='-1',
         maxSize=str(maxSize), startFrame=str(startFrame), __manual=True)
