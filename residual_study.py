import xarray as xr
import numpy as np
import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
# from scipy import stats
# import seaborn as sns

'''
Le script sert à calculer le résidus entre le moléculaire atténué et le signal calibré, puis les représenter en histograme. 
'''

def residus_alt(new, channelnb):
    limite = np.where((new['range']>5)&(new['range']<20))[0]
    residue = np.nansum(new.isel(channel=channelnb, range=limite)['calibrated']-new.isel(channel=channelnb, range=limite)['simulated'], axis=1)
    residues = new.isel(channel=channelnb)['calibrated'] - new.isel(channel=channelnb)['simulated']
    return residue, residues

lidar = 'LI1200'
listfiles = sorted(Path('/homedata/nmpnguyen/OPAR/Processed/RF/'+lidar).glob('*RF_v1.nc'))
### --------------------------------------Create empty array--------------------------------------
r_all = []
time_clouds=np.array([], dtype='datetime64[ns]')
time_clearsky=np.array([], dtype='datetime64[ns]')

### --------------------------------------Separate processing--------------------------------------
for path in listfiles:
    file = xr.open_dataset(path)
    r = residus_alt(file, 1)
    time_clouds = np.concatenate((time_clouds, file['time'][r[0]>0.0005].values))
    time_clearsky = np.concatenate((time_clearsky, file['time'][(r[0]>-0.0005)&(r[0]<0.0005)].values))
    #     r_all = np.concatenate((r_all, r[0][r[0]>0.0005]))   

### --------------------------------------Cloud Hypothesis test--------------------------------------
for t in time_clouds:
    timetofile = pd.to_datetime(t).strftime('%Y-%m-%d')
    filetest = xr.open_dataset(sorted(Path('/homedata/nmpnguyen/OPAR/Processed/RF/'+lidar).glob(timetofile+'RF_v1.nc'))[0])
    limiteZ = (filetest['range']>5) & (filetest['range']<20)
    maxSR = (filetest['calibrated']/filetest['simulated']).sel(time=t).isel(channel=1, range=limiteZ).max()
    if maxSR.values > 1.5:
        print(f'{lidar};{t};2')
    elif (maxSR.values > 0.5) & (maxSR.values < 1.5):
        print(f'{lidar};{t};1')
    else:
        print(f'{lidar};{t};0')
        
### --------------------------------------Clearsky Hypothesis test--------------------------------------
for t in time_clearsky:
    timetofile = pd.to_datetime(t).strftime('%Y-%m-%d')
    filetest = xr.open_dataset(sorted(Path('/homedata/nmpnguyen/OPAR/Processed/RF/'+lidar).glob(timetofile+'RF_v1.nc'))[0])
    limiteZ = (filetest['range']>5) & (filetest['range']<20)
    maxSR = (filetest['calibrated']/filetest['simulated']).sel(time=t).isel(channel=1, range=limiteZ).max()
    if maxSR.values > 1.5:
        print(f'{lidar};{t};2')
    elif (maxSR.values > 0.5) & (maxSR.values < 1.5):
        print(f'{lidar};{t};1')
    else:
        print(f'{lidar};{t};0')

# f, ax = plt.subplots(figsize=[9,6])
# (n, bins, patches )= ax.hist(r_all, bins=100, density=False)
# ax.legend(loc='best', frameon=False)
# # ax.set(ylabel='Density')
# # ax.set_xlim(-0.001, 0.001)    
# ax.set_title(f'Residue between normalized signal and molecular signal \n integral on the altitude\n/homedata/nmpnguyen/OPAR/Processed/RF/LIO3T/clearsky/')
# plt.savefig('/homedata/nmpnguyen/OPAR/Processed/RF/LIO3T/clearsky/Residual_between_normalized_signal_and_molecular_signal.png')
# plt.close()