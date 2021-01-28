# Le script servit à reconstruire molecular ATB from ERA data 

#--------------------------------
# import xarray as xr 
import numpy as np 
import pandas as pd 
import netCDF4 as nc4
from netCDF4 import Dataset
import glob, os
import scipy.interpolate as spi
import numpy as np
from datetime import datetime, timedelta


#read li1200
import glob

geopt = pd.read_pickle("/homedata/nmpnguyen/OPAR/Processed/geopt_final.pkl")
ta = pd.read_pickle("/homedata/nmpnguyen/OPAR/Processed/ta_final.pkl")

# calculate geometric height from ERA5 reanalysis geopotentials(G_era)
lat = np.deg2rad(geopt['latitude'][0])
g0 = 9.80665 #m.s-2
G_era = geopt['geopt']
H_era = G_era/g0 
acc_gravity = 9.78032*(1+5.2885e-3*(np.sin(lat))**2 - 5.9e-6*(np.sin(2*lat))**2)
r0 = 2*acc_gravity/(3.085462e-6 + 2.27e-9*np.cos(2*lat) - 2e-12*np.cos(4*lat))
Z_era = (H_era*(r0))/(acc_gravity*r0/g0 - H_era) # calculer Altitude 
G_era = pd.merge(G_era, Z_era, left_index=True, right_index=True) # merger Geopt et Altitude 
M = 28.966E-3 
R = 8.314510
T = (15 + 273.15)
const = -(M*g0)/(R*T)
p0 = 101325 #pascal Pa
P_era = p0*np.exp(const*Z_era) # calculer Pression en Pa
G_era = pd.merge(G_era, P_era, left_index=True, right_index=True) #merger Geopt et Altitude et Pression
G_era = G_era.rename(columns = {"geopt_x":"geopt", "geopt_y":"altitude", "geopt":"pression"})
era = pd.merge(G_era, ta[['ta']], left_index=True, right_index=True) # merger Geopt et Altitude et Pression et Temperature
#--------------calculer beta mol
k = 1.38e-23
const355 = (5.45e-32/1.38e-23)*((355e-3/0.55)**(-4.09))
const532 = (5.45e-32/1.38e-23)*((532e-3/0.55)**(-4.09))
era['beta355'] = const355*era['pression'].div(era['ta'])
era['beta532'] = const532*era['pression'].div(era['ta'])
era['alpha355'] = era['beta355']/0.119
era['alpha532'] = era['beta532']/0.119
era = era.sort_index()
time = np.unique(era.index.get_level_values(0))
level = np.unique(era.index.get_level_values(1))
era['tau355']=0 ;  era['tau532'] = 0
A = pd.DataFrame()
for t in time:
    a = era.loc[pd.IndexSlice[t,:],:].sort_index(ascending = False)
    #a['tau355'] = 0; a['tau532'] = 0
    for i in range(1, a.shape[0]):
        a['tau355'].iloc[i] = a['tau355'].iloc[i-1] + a['alpha355'].iloc[i]*(a['altitude'].iloc[i]-a['altitude'].iloc[i-1])
        a['tau532'].iloc[i] = a['tau532'].iloc[i-1] + a['alpha532'].iloc[i]*(a['altitude'].iloc[i]-a['altitude'].iloc[i-1])
    A = pd.concat((A, a), axis=0)


A['beta355mol'] = A['beta355']*np.exp(-2*1.7*A['tau355'])
A['beta532mol'] = A['beta532']*np.exp(-2*1.7*A['tau532'])

A.to_pickle("/homedata/nmpnguyen/OPAR/Processed/pression_tempe_backscatter_2019_v3.pkl")




# era = pd.read_pickle("/homedata/nmpnguyen/OPAR/Processed/pression_tempe_backscatter_2019_v2.pkl")
# era_to_nc = era.loc["2019-03-17 00:00:00"]



# #----------------create file netcdf
# ncfile = nc4.Dataset("/homedata/nmpnguyen/OPAR/Processed/Era_for_Opar.nc", mode='w', format = 'NETCDF4')
# alt_dim = ncfile.createDimension('alt', era_to_nc["altitude"].shape[0])
# time_dim = ncfile.createDimension('time', 1)
# print(ncfile)
# ncfile.title = "Era_for_Opar 2019-03-17 00:00:00"
# ncfile.subtitle = 'era data'
# alt = ncfile.createVariable('alt', np.float64, ('alt',))
# alt.units = 'm'
# alt.long_name = 'altitude'
# alt[:] = era_to_nc["altitude"]
# # time = ncfile.createVariable('time', np.float64, ('time',))
# # time.units = 'seconds since 2000-01-01'
# # time.long_name = 'time'
# # time.calendar = 'proleptic_gregorian'
# pres = ncfile.createVariable('pression', np.float64, ('alt'))
# pres.units = "Pa"
# pres.standard_name = 'Pression'
# pres[:] = era_to_nc["pression"]

# temp = ncfile.createVariable('tempe', np.float64, ('alt'))
# temp.units = "K"
# temp.standard_name = "Temperature"
# temp[:] = era_to_nc["ta"]

# geopt = ncfile.createVariable('geopt', np.float64, ('alt'))
# geopt.units = "m**2/s**2"
# geopt.standard_name = "Geopotential height"
# geopt[:] = era_to_nc['geopt']



# plot à Vérification: 
# f, ax = plt.subplots(1,2, sharey=True)
# ax[0].plot(era_to_nc['pression'], era_to_nc['altitude'])
# ax[0].set(xlabel="pression, Pa", ylabel="altitude, m")

# ax[1].plot(era_to_nc['ta'], era_to_nc['altitude'])
# ax[1].set(xlabel= "Temperature, K")
# plt.suptitle("2019-03-17 00:00:00")
# plt.savefig("/home/nmpnguyen/PTz_ref.png")

# f, ax = plt.subplots(1,4, sharey=True)
# ax[0].plot(era_to_nc['beta532'], era_to_nc['altitude'], label = "532nm")
# ax[0].plot(era_to_nc['beta355'], era_to_nc['altitude'], label = "355nm")
# ax[0].set(xlabel="ATB, \nm-1.sr-1")
# ax[0].legend()
# ax[1].plot(era_to_nc['alpha532'], era_to_nc['altitude'], label = "532nm")
# ax[1].plot(era_to_nc['alpha355'], era_to_nc['altitude'], label = "355nm")
# ax[1].set(xlabel="alpha, \n ")
# ax[1].legend()
# ax[2].plot(era_to_nc['tau532'], era_to_nc['altitude'], label = "532nm")
# ax[2].plot(era_to_nc['tau355'], era_to_nc['altitude'], label = "355nm")
# ax[2].set(xlabel="tau, \nm-1.sr-1")
# ax[2].legend()
# ax[3].plot(era_to_nc['beta532mol'], era_to_nc['altitude'], label = "532nm")
# ax[3].plot(era_to_nc['beta355mol'], era_to_nc['altitude'], label = "355nm")
# ax[3].set(xlabel="ATBmol, \nm-1.sr-1")
# ax[3].legend()
# plt.suptitle("2019-03-17 00:00:00")
# plt.tight_layout()
# plt.savefig("/home/nmpnguyen/BAT_ref.png")

