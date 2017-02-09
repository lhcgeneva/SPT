xls_db = pd.read_excel(xls_file)
if np.isnan(xls_db['Threshold'][movieNo]):
    xls_db.loc[movieNo, 'Threshold'] = 900
thresh = int(xls_db['Threshold'][movieNo])
if np.isnan(xls_db['StartFrame'][movieNo]):
    xls_db.loc[movieNo, 'StartFrame'] = 0
startFrame = int(xls_db.loc[movieNo, 'StartFrame'])


def adj_thresh(calibFrame, thresh, startFrame):
    calibFrame = int(calibFrame)
    thresh = int(thresh)
    startFrame = int(startFrame)
    o = DiffusionFitter(xls_db['Folder'][movieNo], thresh, pixelSize=0.120,
                        startFrame=startFrame)
    print(xls_db['Folder'][movieNo])
    o.plot_calibration(calibrationFrame=calibFrame)
    xls_db.loc[movieNo, 'StartFrame'] = startFrame
    xls_db.loc[movieNo, 'Threshold'] = thresh
    xls_db.to_excel(xls_file)


interact(adj_thresh, thresh=str(thresh), calibFrame='200',
         startFrame=str(startFrame), __manual=True)
