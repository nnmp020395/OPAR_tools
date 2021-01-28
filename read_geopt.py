import xarray as xr 
import numpy as np 
import pandas as pd 
import netCDF4 as nc4
import matplotlib.pyplot as plt
import glob, os
from netCDF4 import Dataset

# geopotentiel
# get position of longitude and latitude 
# position_ref = [55.00, -21.00]
# dir_geopt = os.path.join("/bdd/ERA5/NETCDF/GLOBAL_025/4xdaily/AN_PL/2019/geopt.201907.aphe5.GLOBAL_025.nc")
# lon = xr.open_dataset(dir_geopt).longitude.values
# lat = xr.open_dataset(dir_geopt).latitude.values
# # ajouter une selection du temps + boucle sur chaque temps avant de selectionner longitude /latitude
# position_index = [np.where(lon == position_ref[0])[0][0], np.where(lat == position_ref[1])[0][0]]
# g = xr.open_dataset(dir_geopt).geopt
# gdt = g.isel(longitude = position_index[0], latitude = position_index[1])
# gdt.to_netcdf("/home/nmpnguyen/geopt_test.nc", mode="w", format="NETCDF4")
# dgeopt = xr.open_mfdataset("/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/2019/geopt*.nc", parallel=True, chunks = {'time':100})
# # geopt = dgeopt.geopt
# geopt1 = dgeopt.isel(longitude = position_index[0], latitude = position_index[1])
# geopt_dset = geopt1.to_dataset()
# # to_netcdf()

# # temparature
# # get position of longitude and latitude 
# dir_ta = os.path.join("/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/2019/ta.201907.ap1e5.GLOBAL_025.nc")
# lon = xr.open_dataset(dir_ta).longitude.values
# lat = xr.open_dataset(dir_ta).latitude.values
# position_index = [np.where(lon == position_ref[0])[0][0], np.where(lat == position_ref[1])[0][0]]
# dta = xr.open_mfdataset("/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/2019/ta*.nc", parallel=True, chunks = {'time':100})
# # ta = dta.ta.values
# ta1 = dta.isel(longitude = position_index[0], latitude = position_index[1])
# # ta_dset = ta1.to_dataset()


position_ref = [55.00, -21.00]
geopt_final = pd.DataFrame()
files = []
for file in glob.glob("/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/2018/ta*.nc"):
    files.append(file)

for i in range(0, len(files)):
    dir_geopt = files[i]
    print(dir_geopt)
    lon = xr.open_dataset(dir_geopt).longitude.values
    lat = xr.open_dataset(dir_geopt).latitude.values   
    position_index = [np.where(lon == position_ref[0])[0][0], np.where(lat == position_ref[1])[0][0]]
    g = xr.open_dataset(dir_geopt).ta
    geopt = g.isel(longitude = position_index[0], latitude = position_index[1])
    df_geopt = geopt.to_dataframe()
    geopt_final = pd.concat([geopt_final, df_geopt])

geopt_final.to_pickle("/homedata/nmpnguyen/ta_final_2018.pkl")

# --------------------------------------------------------------------

# geopt = pd.read_pickle("/homedata/nmpnguyen/OPAR/Processed/geopt_final.pkl")
# geopt = geopt.reset_index()
# geopt = geopt.set_index('time')
# geopt['day'] = geopt.index.day
# geopt['month'] = geopt.index.month
# geopt['year'] = geopt.index.year
# geopt['hour'] = geopt.index.hour
# geopt['altitude'] = geopt.geopt/9.8

# M = 28.966E-3 
# g = 9.805
# R = 8.314510
# T = (15 + 273.15)
# const = -(M*g)/(R*T)
# p0 = 101325 #pascal Pa
# # geopt[['pression']].iloc[-1] = p0*np.exp(const*geopt[['altitude']].iloc[-1])
# p1 = p0*np.exp(const*geopt[['altitude']].iloc[-1])
# geopt['pression'] = p1*np.exp(const*(geopt[['altitude']][:] - geopt[['altitude']].iloc[-1]))
# geopt.to_pickle("/homedata/nmpnguyen/OPAR/pression.pkl")


# ta = pd.read_pickle("/homedata/nmpnguyen/OPAR/ta_final.pkl")
# ta = ta.reset_index()
# ta = ta.set_index('time')


# geopt = pd.read_pickle("/homedata/nmpnguyen/OPAR/geopt_final.pkl")
# ta = pd.read_pickle("/homedata/nmpnguyen/OPAR/ta_final.pkl")
# data = pd.merge(geopt, ta[['ta']], left_index=True, right_index=True)
# data['altitude'] = data.geopt/9.8
# geopt['altitude'] = geopt.geopt/9.8
# M = 28.966E-3 
# g = 9.805
# R = 8.314510
# T = (15 + 273.15)
# const = -(M*g)/(R*T)
# p0 = 101325 
# p1 = p0*np.exp(const*geopt[['altitude']].iloc[-1])
# altitude1 = geopt[['altitude']].iloc[-1]
# data['pression'] = p1*np.exp(const*(data[['altitude']][:] - altitude1))

# k = 1.38E-23
# const1 = 5.45E-32*(355E-3/0.55)**(-4.09)
# data['beta355'] = (data['pression'][:]/(k*data['ta'][:]))*const1
# const2 = 5.45E-32*(532E-3/0.55)**(-4.09)
# data['beta532'] = (data['pression'][:]/(k*data['ta'][:]))*const2

# data = data.reset_index()
# data = data.set_index('time')
# li1200['interval_time'] = li1200.index.strftime("%Y-%m-%d %H:00")

# data['day'] = data.index.day
# data['month'] = data.index.month
# data['year'] = data.index.year
# data['hour'] = data.index.hour


# li1200 = pd.read_pickle("/homedata/nmpnguyen/OPAR/Processed/LI1200/li1200_processed.pkl")
# li1200 = li1200.set_index([li1200.columns[0], 'time'])
# li1200['interval_time'] = li1200.index.strftime("%Y-%m-%d %H:00:00")

# li1200.loc[li1200.index.hour.isin([0,1,2,3,4,5]), 'interval_time'] = li1200[li1200.index.hour.isin([0,1,2,3,4,5])].index.strftime("%Y-%m-%d 00:00:00")
# li1200.loc[li1200.index.hour.isin([6,7,8,9,10,11]), 'interval_time'] = li1200[li1200.index.hour.isin([6,7,8,9,10,11])].index.strftime("%Y-%m-%d 06:00:00")
# li1200.loc[li1200.index.hour.isin([12,13,14,15,16,17]), 'interval_time'] = li1200[li1200.index.hour.isin([12,13,14,15,16,17])].index.strftime("%Y-%m-%d 12:00:00")
# li1200.loc[li1200.index.hour.isin([18,19,20,21,22,23]), 'interval_time'] = li1200[li1200.index.hour.isin([18,19,20,21,22,23])].index.strftime("%Y-%m-%d 18:00:00")


# 23,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22