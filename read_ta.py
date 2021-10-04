import xarray as xr 
import numpy as np 
import pandas as pd 
# import netCDF4 as nc4
# import matplotlib.pyplot as plt
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
for file in glob.glob("/bdd/ERA5/NETCDF/GLOBAL_025/4xdaily/AN_PL/2019/ta*.nc"):
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

geopt_final.to_pickle("/homedata/nmpnguyen/ta_final.pkl")





