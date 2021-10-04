import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def residus_alt(new, channelnb):
    limite = np.where((new['range']>5)&(new['range']<20))[0]
    residues = new.isel(channel=channelnb, range=limite)['calibrated'] - new.isel(channel=channelnb, range=limite)['simulated']
    residue = np.nansum(residues, axis=1)
    code = np.zeros_like(residue)
    code[residue>0.0005] = 2
    code[(residue>-0.0005)&(residue<0.0005)] = 1
    return code

'''
Préparation des données de LIO3T 532nm
'''

listfiles = sorted(Path('/homedata/nmpnguyen/OPAR/Processed/RF/LIO3T/clearsky/').glob('2020*RF_v1.nc'))

loaded = xr.open_dataset(listfiles[0])
limite = np.where((loaded['range']*1e3>loaded.attrs['calibration height'][0])&
                  (loaded['range']*1e3<20000))[0]
flagcode = residus_alt(loaded, channelnb=0)
atb532 = xr.open_dataset(listfiles[0]).isel(channel=0, range=limite)['calibrated']
sr532 = (loaded['calibrated']/loaded['simulated']).isel(time=np.where(flagcode!=0)[0], channel=0, range=limite)

for file in listfiles[1:]:
    loaded = xr.open_dataset(file)
    print(file)
    flagcode = residus_alt(loaded, channelnb=0)
    # atb532 = xr.concat([atb532, loaded.isel(channel=0, range=limite)['calibrated']], dim='time')
    sr532 = xr.concat([sr532, (loaded['calibrated']/loaded['simulated']).isel(time=np.where(flagcode!=0)[0], channel=0, range=limite)], dim='time')
    
atb532mean = atb532.resample(time='15min').mean()
atb532mean_reshape = np.mean(np.reshape(np.array(atb532mean), (atb532mean.shape[0], int(atb532mean.shape[1]/2), 2)), axis=2)
# np.round(np.array(atb532mean['range']).reshape(-1, 2).mean(axis=1), 2)
sr532mean = sr532.resample(time='15min').mean()
sr532mean_reshape = np.mean(np.reshape(np.array(sr532mean), (sr532mean.shape[0], int(sr532mean.shape[1]/2), 2)), axis=2)

print(f'nombre des jours de mesures en 532nm: {len(listfiles)}')
nb_profils = 0
for file in listfiles:
    loaded = xr.open_dataset(file)
    nb_profils = nb_profils + len(loaded['time'])

print(f'nombre des profils de mesures en 532nm: {nb_profils}')
footnote532 = f'532nm: {len(listfiles)} jours, {nb_profils} profils'

'''
Préparation des données de LI1200 355nm
'''

listfiles = sorted(Path('/homedata/nmpnguyen/OPAR/Processed/RF/LI1200/clearsky/').glob('2020*RF_v1.nc'))

loaded = xr.open_dataset(listfiles[0])
limite = np.where((loaded['range']*1e3>loaded.attrs['calibration height'][0])&
                 (loaded['range']*1e3<20000))[0]
flagcode = residus_alt(loaded, channelnb=1)
atb355 = xr.open_dataset(listfiles[0]).isel(channel=1, range=limite)['calibrated']
sr355 = (loaded['calibrated']/loaded['simulated']).isel(time=np.where(flagcode!=0)[0], channel=1, range=limite)

for file in listfiles[1:]:
    loaded = xr.open_dataset(file)
    print(file)
    flagcode = residus_alt(loaded, channelnb=1)
    # atb355 = xr.concat([atb355, loaded.isel(channel=1, range=limite)['calibrated']], dim='time')
    sr355 = xr.concat([sr355, (loaded['calibrated']/loaded['simulated']).isel(time=np.where(flagcode!=0)[0], channel=0, range=limite)], dim='time')

atb355mean = atb355.resample(time='15min').mean()
sr355mean = sr355.resample(time='15min').mean()
# atb355mean_reshape = np.mean(np.reshape(np.array(atb355mean[:,:]), (atb355mean.shape[0], int(atb355mean.shape[1]/2), 2)), axis=2)
# sr355mean_reshape = np.mean(np.reshape(np.array(sr355mean[:,:]), (sr355mean.shape[0], int(atb355mean.shape[1]/2), 2)), axis=2)

print(f'nombre des jours de mesures en 355nm: {len(listfiles)}')
nb_profils = 0
for file in listfiles:
    loaded = xr.open_dataset(file)
    nb_profils = nb_profils + len(loaded['time'])

print(f'nombre des profils de mesures en 355nm: {nb_profils}')
footnote355 = f'355nm: {len(listfiles)} jours, {nb_profils} profils'

'''
PLOT + LINEAR FIT LINE
'''
xy, xids, yids = np.intersect1d(np.round(np.array(sr532mean['range']).reshape(-1, 2).mean(axis=1), 2), 
              np.round(np.array(sr355mean['range']),2),
              return_indices = True)
time_xy, time_xids, time_yids = np.intersect1d(sr532mean['time'].values, sr355mean['time'].values,
                                              return_indices=True)

X532 = sr532mean_reshape[:, xids]
X532 = X532[time_xids,:]

Y355 = np.array(sr355mean)[:, yids]
Y355 = Y355[time_yids,:]

ids = np.intersect1d(np.where(~np.isnan(X532.ravel()))[0], np.where(~np.isnan(Y355.ravel()))[0])
      
H = np.histogram2d(X532.ravel()[ids], Y355.ravel()[ids], bins=100,
                  range = [[0, 100], [0, 100]])
Hprobas = H[0]/len(X532.ravel())*100
Xxedges, Yyedges = np.meshgrid(H[1], H[2])
    
print(f'nombre des points SR en 532nm: {len(X532.ravel()[ids])}')
print(f'nombre des points SR en 355nm: {len(Y355.ravel()[ids])}')
footnote355 = footnote355 + f'{len(Y355.ravel()[ids])} points'
footnote532 = footnote532 + f'{len(X532.ravel()[ids])} points'

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from matplotlib.colors import LogNorm
from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(X532.ravel()[ids], Y355.ravel()[ids])
fitLine = slope * X532.ravel()[ids] + intercept

ff, (ax, axins) = plt.subplots(figsize=[6,10], nrows=2)
p = ax.pcolormesh(Xxedges, Yyedges, Hprobas.T, norm = LogNorm(vmax=1e0, vmin=1e-6)) #
c = plt.colorbar(p, ax=ax, label='%')
ax.set(xlabel='SR532', ylabel='SR355', 
       title=f'OPAR, average each 15 mins x 15m \n LinearRegression: {round(slope,5)}x + {round(intercept,3)}')
# ax.plot(X532.ravel()[ids], fitLine, c='r') #Fit line
ax.set_xlim(0,100)
ax.set_ylim(0,100)

pins = axins.pcolormesh(Xxedges, Yyedges, Hprobas.T, norm = LogNorm(vmax=1e-02, vmin=1e-6))
cins = plt.colorbar(pins, ax=axins, label='%')
axins.set_ylim(0, 20)
axins.set_xlim(0, 20)
axins.set(xlabel='SR532', ylabel='SR355')
# Add a footnote below and to the right side of the chart
axins.annotate(footnote355+'\n'+footnote532,
            xy = (1.0, -0.2),
            xycoords='axes fraction',
            ha='right',
            va="center",
            fontsize=10)
ff.tight_layout()
plt.savefig(Path('/homedata/nmpnguyen/OPAR/Processed/RF/', 'distributionSR_100x100_OPAR2020_clearsky.png'))
