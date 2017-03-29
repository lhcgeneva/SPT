xls_db['IntervalReal'] = 'nan'
xls_db['ExposureReal'] = 'nan'
xls_db['Length'] = 'nan'
for i, path in enumerate(xls_db['Folder']):
    p = ParticleFinder(path)
    xls_db.loc[i, 'IntervalReal'] = p.timestep
    xls_db.loc[i, 'ExposureReal'] = p.exposure
    xls_db.loc[i, 'Length'] = len(p.frames)
xls_db.to_excel(xls_file)
