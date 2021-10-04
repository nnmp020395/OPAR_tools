import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pathlib import Path
from matplotlib.colors import LogNorm
from scipy import stats

plt.rcParams['agg.path.chunksize'] = 10000
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
# from matplotlib.colors import LogNorm

'''
Etape 1: En déduire la distribution des SR
'''
# listfiles = sorted(Path('/homedata/nmpnguyen/IPRAL/RF/Calibrated/').glob('*2018*_1440.nc'))

# loaded = xr.open_dataset(listfiles[0])
# limite = np.where((loaded['range']>5000)&
#                   (loaded['range']<20000))[0]
# atb355 = xr.open_dataset(listfiles[0]).isel(wavelength=0, range=limite)['calibrated']
# sr355 = loaded.isel(wavelength=0, range=limite)['calibrated']/loaded.isel(wavelength=0, range=limite)['simulated']
# atb532 = xr.open_dataset(listfiles[0]).isel(wavelength=1, range=limite)['calibrated']
# sr532 = loaded.isel(wavelength=1, range=limite)['calibrated']/loaded.isel(wavelength=1, range=limite)['simulated']

# for file in listfiles[1:]:
#     loaded = xr.open_dataset(file)
#     print(file)
# #     atb532 = xr.concat([atb355, loaded.isel(channel=1, range=limite)['calibrated']], dim='time')
#     sr355 = xr.concat([sr355, loaded.isel(wavelength=0, range=limite)['calibrated']/loaded.isel(wavelength=0, range=limite)['simulated']], dim='time')
#     sr532 = xr.concat([sr532, loaded.isel(wavelength=1, range=limite)['calibrated']/loaded.isel(wavelength=1, range=limite)['simulated']], dim='time')

# sr532mean = sr532#.resample(time='15min').mean()
# sr355mean = sr355#.resample(time='15min').mean()
# sr355H = sr355mean.values.flatten()
# sr532H = sr532mean.values.flatten()

'''
Etape 1+: En déduire la distribution des SR
'''
listfiles = sorted(Path('/homedata/nmpnguyen/IPRAL/RF/Calibrated/noproblem/').glob('*2018*_1440.nc'))
sr_limite = [60,20] #SR532~60 et SR355~20
sr532H = np.array([], dtype=np.float)
sr355H = np.array([], dtype=np.float)
alltimes = np.array([], dtype='datetime64[ns]')

for file in listfiles:
    print(file)
    loaded = xr.open_dataset(file)
    limite = np.where((loaded['range']>5000)&(loaded['range']<14000))[0]
    sr532 = (loaded.isel(wavelength=1, range=limite)['calibrated']/loaded.isel(wavelength=1, range=limite)['simulated'])#.values.flatten()
#     print(np.argmax(sr532.values, axis=0))
    sr355 = (loaded.isel(wavelength=0, range=limite)['calibrated']/loaded.isel(wavelength=0, range=limite)['simulated'])#.values.flatten()
#     print(len(loaded['time'].values),len(np.argmax(sr355.values, axis=1)),np.argmax(sr355.values, axis=1))
    # ids = ~np.isinf(sr355) & ~np.isinf(sr532)
    # ids = (sr355==sr_limite[1])&(sr532==sr_limite[0])
    sr355H = xr.concat((sr355H, sr355), dim='time')
    sr532H = xr.concat((sr532H, sr532), axis=0)    
    alltimes = np.concatenate((alltimes, loaded['time'].values))

# nb_profils = 0
# for file in listfiles:
#     loaded = xr.open_dataset(file)
#     nb_profils = nb_profils + len(loaded['time'])
# print(f'nombre des jours: {len(listfiles)}')
# print(f'nombre des profiles: {nb_profils}')   
# print(f'nombre de points: {len(sr355H)}')
# footnote = f'{len(listfiles)} jours, {nb_profils} profiles, {len(sr355H)} points'

# H = np.histogram2d(sr532H, sr355H, bins=100, range = [[0, sr_limite], [0, sr_limite]]) 
# Hprobas = H[0]/len(sr355H)*100
# Xxedges, Yyedges = np.meshgrid(H[1], H[2])

# mask = ~np.isnan(sr532H) & ~np.isnan(sr355H)
# with np.errstate(invalid='ignore'):
#     slope, intercept, r_value, p_value, std_err = stats.linregress(sr532H[mask], sr355H[mask])
    
# fitLine = slope * sr532H + intercept

# ff, (ax, axins) = plt.subplots(figsize=[6,10], nrows=2)
# p = ax.pcolormesh(Xxedges, Yyedges, Hprobas.T, norm = LogNorm(vmax=1e0, vmin=1e-6))
# # ax.plot(sr532H, fitLine, '-.', c='r')
# c = plt.colorbar(p, ax=ax, label='%')
# ax.set(xlabel='SR532', ylabel='SR355', 
#        title= f'IPRAL, vert.rezolution = 15m \nLinearRegression: {round(slope,5)}x + {round(intercept,3)}')
# ax.set(xlim=(0,sr_limite), ylim=(0,sr_limite))

# pins = axins.pcolormesh(Xxedges, Yyedges, Hprobas.T, norm = LogNorm(vmax=1e-02, vmin=1e-6))
# cins = plt.colorbar(pins, ax=axins, label='%')
# axins.set_ylim(0, 20)
# axins.set_xlim(0, 20)
# axins.set(xlabel='SR532', ylabel='SR355')
# # Add a footnote below and to the right side of the chart
# axins.annotate('2018'+footnote,
#             xy = (1.0, -0.2),
#             xycoords='axes fraction',
#             ha='right',
#             va="center",
#             fontsize=10)
# ff.tight_layout()
# plt.savefig(Path('/homedata/nmpnguyen/IPRAL/RF/Calibrated/','distributionSR_100x100_Ipral2018_noproblem.png'))


### Save sr355 and sr532 
ds = xr.Dataset(data_vars = {"SR532": (("time","range"), sr532H),
                                "SR355": (("time","range"), sr355H),}, 
                coords = {
                        "time": alltimes,
                        "range": loaded['range'][limite].values,                     
                    },)
ds.to_netcdf('/homedata/nmpnguyen/IPRAL/SR.nc', 'w')