import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pathlib import Path
from matplotlib.colors import LogNorm

'''
Etape 3: En déduire la distribution des SR
'''
sr532H = np.array([], dtype=np.float)
sr355H = np.array([], dtype=np.float)

for file in listfiles:
    loaded = xr.open_dataset(file)
#     print(loaded)
    limite = np.where((loaded['range']>5000)&(loaded['range']<20000))[0]
    sr532 = (loaded.isel(wavelength=1, range=limite)['calibrated']/loaded.isel(wavelength=1, range=limite)['simulated']).values.flatten()
    sr355 = (loaded.isel(wavelength=0, range=limite)['calibrated']/loaded.isel(wavelength=0, range=limite)['simulated']).values.flatten()
    ids = np.intersect1d(np.where(~np.isnan(sr355))[0], np.where(~np.isnan(sr532))[0])
    sr532H = np.concatenate((sr532H, sr532[ids]))
    sr355H = np.concatenate((sr355H, sr355[ids]))
    

from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(sr532H, sr355H)    
fitLine = slope * sr532H + intercept