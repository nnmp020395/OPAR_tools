"""
Script is used to create Simulated Molecular Attenuated Backscatter from Pression and Temperature of ERA5 database 
"""

import xarray as xr 
from pathlib import Path
import numpy as np 
import pandas as pd 
import netCDF4 as nc4
import glob, os
# import scipy.interpolate as spi
from datetime import datetime, timedelta
import matplotlib.dates as plt_dates 
# from matplotlib.backends.qt_compat import QtGui
import matplotlib.pyplot as plt
from argparse import Namespace, ArgumentParser


__author__ = "N-M-Phuong Nguyen"
__version__ = "0.1.3"


def simulate_betamol(path_raw, lidar_name, era):
    path_raw = Path(path_raw)
    print("------------Read opar data file------------")
    print(f"Reading Opar data file {path_raw.name}")
    day = path_raw.name.split(".")[0]
    li1200 = nc4.Dataset(path_raw, 'r')
    rangeLi = li1200.variables['range'][:]
    altLi = rangeLi + 2.1
    zli1200 = altLi*1000
    # read firtly data under 5km
    rSelectLi = np.array(rangeLi[np.where(altLi < 25)]) #en km
    zSelectLi = np.array(zli1200[np.where(altLi < 25)]) #em m
    print("------------Read ERA5 data file------------")
    eraSelect = era.loc[day+" 00:00:00":day+" 18:00:00"]#.loc[era['altitude'] < 5000]
    #----------
    print("------------Read ERA5 data file------------")
    time, calendar, units_time = li1200.variables['time'][:], li1200.variables['time'].calendar, li1200.variables['time'].units
    timeLi = pd.to_datetime(nc4.num2date(time, units_time, calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True))
    timeEra = timeLi.to_numpy(copy = True) 
    #----------
    timeEra[np.where(timeLi.hour.isin([0,1,2,3,4,5]))] = np.array(day+" 00:00:00").astype("datetime64[s]")
    timeEra[np.where(timeLi.hour.isin([6,7,8,9,10,11]))] = np.array(day+" 06:00:00").astype("datetime64[s]")
    timeEra[np.where(timeLi.hour.isin([12,13,14,15,16,17]))] = np.array(day+" 12:00:00").astype("datetime64[s]")
    timeEra[np.where(timeLi.hour.isin([18,19,20,21,22,23]))] = np.array(day+" 18:00:00").astype("datetime64[s]")
    #----------
    levels = np.unique([el[1] for el in eraSelect.index]) # altitude level of era data 
    df_index = pd.MultiIndex.from_product([timeLi, levels], names = ['timeLidar', 'level']) # index time of opar data + index altitude of era data
    new_index = pd.MultiIndex.from_product([timeLi, zSelectLi], names = ['timeLidar', 'z']) # index time of opar data et index altitude op opar data also
    columns_names = ['altitude', 'pression', 'ta', 'beta355', 'beta532', 'alpha355', 'alpha532', 'tau355', 'tau532', 'beta355mol', 'beta532mol'] #
    df = pd.DataFrame(index = df_index, columns = columns_names) # variable where stock simulated data 
    # interpolation -> simulated molecular ATB from ERA
    columns_names_interp = [str(cn) + "_interp" for cn in columns_names]
    beta355_interp ,beta532_interp ,tau355_interp ,tau532_interp ,alpha355_interp ,alpha532_interp ,beta355mol_interp ,beta532mol_interp, pression_interp, ta_interp = [[] for _ in range(len(columns_names_interp)-1)]
    for el, ell in zip(timeEra, timeLi):
        # print(el, ell)
        for l in levels:
            df.loc[ell, l][columns_names] = eraSelect.loc[el, l][columns_names]
        f1 = spi.interp1d(df.loc[ell,'altitude'], df.loc[ell,'beta355mol'], kind = 'linear', bounds_error=False) #  fill_value = -9999., 
        f2 = spi.interp1d(df.loc[ell,'altitude'], df.loc[ell,'beta532mol'], kind = 'linear', bounds_error=False)
        f3 = spi.interp1d(df.loc[ell,'altitude'], df.loc[ell,'tau355'], kind = 'linear', bounds_error=False)
        f4 = spi.interp1d(df.loc[ell,'altitude'], df.loc[ell,'tau532'], kind = 'linear', bounds_error=False)
        f5 = spi.interp1d(df.loc[ell,'altitude'], df.loc[ell,'alpha355'], kind = 'linear', bounds_error=False)
        f6 = spi.interp1d(df.loc[ell,'altitude'], df.loc[ell,'alpha532'], kind = 'linear', bounds_error=False)
        f7 = spi.interp1d(df.loc[ell,'altitude'], df.loc[ell,'beta355'], kind = 'linear', bounds_error=False)
        f8 = spi.interp1d(df.loc[ell,'altitude'], df.loc[ell,'beta532'], kind = 'linear', bounds_error=False)
        f9 = spi.interp1d(df.loc[ell,'altitude'], df.loc[ell,'pression'], kind = 'linear', bounds_error=False)
        f10 = spi.interp1d(df.loc[ell,'altitude'], df.loc[ell,'ta'], kind = 'linear', bounds_error=False)
        beta355mol_interp, beta532mol_interp = np.append(beta355mol_interp, np.array(f1(zSelectLi))), np.append(beta532mol_interp, np.array(f2(zSelectLi)))
        tau355_interp, tau532_interp = np.append(tau355_interp, np.array(f3(zSelectLi))), np.append(tau532_interp, np.array(f4(zSelectLi)))
        alpha355_interp, alpha532_interp = np.append(alpha355_interp, np.array(f5(zSelectLi))), np.append(alpha532_interp, np.array(f6(zSelectLi)))
        beta355_interp, beta532_interp = np.append(beta355_interp, np.array(f7(zSelectLi))), np.append(beta532_interp, np.array(f8(zSelectLi)))
        pression_interp, ta_interp = np.append(pression_interp, np.array(f9(zSelectLi))), np.append(ta_interp, np.array(f10(zSelectLi)))
    # save simulated molecular ATB file 
    new_df = pd.DataFrame(index = new_index, data = np.array([pression_interp, ta_interp, beta355_interp ,beta532_interp ,alpha355_interp ,alpha532_interp ,tau355_interp ,tau532_interp ,beta355mol_interp ,beta532mol_interp]).T, columns = columns_names[1:])
    new_df.to_pickle("/homedata/nmpnguyen/OPAR/Processed/"+lidar_name.upper()+"/"+day+"_simul.pkl")
    to_netcdf_simuOPAR(day, lidar_name)

def to_netcdf_simuOPAR(fn, lidar_name):
    path_simu = "/homedata/nmpnguyen/OPAR/Processed/"+lidar_name.upper()+"/"+fn+"_simul.pkl"
    file_to_nc = pd.read_pickle(path_simu)
    #--------------------
    ncfile = nc4.Dataset("/homedata/nmpnguyen/OPAR/Processed/"+lidar_name.upper()+"/"+fn+"_simul.nc", mode='w', format = 'NETCDF4')
    #--------------------
    alt_dim = ncfile.createDimension('alt', np.unique(file_to_nc.index.get_level_values(1)).shape[0])
    time_dim = ncfile.createDimension('time', np.unique(file_to_nc.index.get_level_values(0)).shape[0])
    print(ncfile)
    ncfile.title = "simulated profile for OPAR measured data"
    ncfile.subtitle = lidar_name.upper()+" "+fn 
    #--------------------
    alt = ncfile.createVariable('alt', np.float64, ('alt',))
    alt.units = 'm'
    alt.long_name = 'altitude'
    alt[:] = np.unique(file_to_nc.index.get_level_values(1))
    #--------------------
    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = 'seconds since 2000-01-01'
    time.long_name = 'time'
    time.calendar = 'proleptic_gregorian'
    time[:] = np.unique(file_to_nc.index.get_level_values(0))
    #--------------------
    pres = ncfile.createVariable('pression', np.float64, ('time','alt'))
    pres.units = "Pa"
    pres.standard_name = 'Pression'
    pres[:] = file_to_nc["pression"].unstack(level=1)
    #--------------------
    temp = ncfile.createVariable('tempe', np.float64, ('time','alt'))
    temp.units = "K"
    temp.standard_name = "Temperature"
    temp[:] = file_to_nc["ta"].unstack(level=1)
    #--------------------
    beta355 = ncfile.createVariable('beta355', np.float64, ('time','alt'))
    beta355.units = "m-1.sr-1"
    beta355.standard_name = "Atmospheric backscatter coefficient"
    beta355[:] = file_to_nc["beta355"].unstack(level=1)
    #--------------------
    alpha355 = ncfile.createVariable('alpha355', np.float64, ('time','alt'))
    alpha355.units = ""
    alpha355.standard_name = "Atmospheric Extinction coefficient"
    alpha355[:] = file_to_nc["alpha355"].unstack(level=1)
    #--------------------
    tau355 = ncfile.createVariable('tau355', np.float64, ('time','alt'))
    tau355.units = ""
    tau355.standard_name = ""
    tau355[:] = file_to_nc["tau355"].unstack(level=1)
    #--------------------
    mol355 = ncfile.createVariable('mol355', np.float64, ('time','alt'))
    mol355.units = "m-1.sr-1"
    mol355.standard_name = "Molecular Backscatter coefficient"
    mol355[:] = file_to_nc["beta355mol"].unstack(level=1)
    #--------------------
    beta532 = ncfile.createVariable('beta532', np.float64, ('time','alt'))
    beta532.units = "m-1.sr-1"
    beta532.standard_name = "Atmospheric backscatter coefficient"
    beta532[:] = file_to_nc["beta532"].unstack(level=1)
    #--------------------
    alpha532 = ncfile.createVariable('alpha532', np.float64, ('time','alt'))
    alpha532.units = ""
    alpha532.standard_name = "Atmospheric Extinction coefficient"
    alpha532[:] = file_to_nc["alpha532"].unstack(level=1)
    #--------------------
    tau532 = ncfile.createVariable('tau532', np.float64, ('time','alt'))
    tau532.units = ""
    tau532.standard_name = ""
    tau532[:] = file_to_nc["tau532"].unstack(level=1)
    #--------------------
    mol532 = ncfile.createVariable('mol532', np.float64, ('time','alt'))
    mol532.units = "m-1.sr-1"
    mol532.standard_name = "Molecular Backscatter coefficient"
    mol532[:] = file_to_nc["beta532mol"].unstack(level=1)
    return ncfile


def variables_from_era(opar_file):
    """
    Le script permet de lire input ERA5 et outpt Pression/Temperature
    input = ipral file path
    
    """
    print('-----GET IPRAL BCK CORRECTED FILE-----')
    d = xr.open_dataset(opar_file)
    time = d.time.values
    YEAR = pd.to_datetime(time[0]).strftime('%Y')
    MONTH = pd.to_datetime(time[0]).strftime('%m')
    lon_opar = round(4*float(d['signal'].longitude))/4 #round(float(d.geospatial_lon_min),2)
    lat_opar = round(4*float(d['signal'].latitude))/4
    print(f'longitude: {lon_opar}')
    print(f'latitude: {lat_opar}')
    #----
    print('-----GET ERA5 FILE-----')
    ERA_FOLDER = Path("/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL")
    ERA_FILENAME = YEAR+MONTH+".ap1e5.GLOBAL_025.nc"
    GEOPT_PATH = ERA_FOLDER / YEAR / Path("geopt."+ERA_FILENAME)
    TA_PATH = ERA_FOLDER / YEAR / Path("ta."+ERA_FILENAME)
    print(f'path of temperature {TA_PATH}')
    print(f'path of geopotential {GEOPT_PATH}')
    geopt = xr.open_dataset(GEOPT_PATH)
    ta = xr.open_dataset(TA_PATH)
    #----
    print('-----CONVERT TIME AND LOCALISATION-----')
    # date_start = pd.to_datetime(time[0])
    # date_end = pd.to_datetime(time[-1])
    time = pd.to_datetime(time).strftime('%Y-%m-%dT%H:00:00.000000000')
    time = time.astype('datetime64[ns]')
    time_unique = np.unique(time)
    LAT = geopt.latitude[np.where(np.abs(geopt.latitude.values - lat_opar) <=0.25)[0][1]].values
    LON = geopt.longitude[np.where(np.abs(geopt.longitude.values - lon_opar) <=0.25)[0][1]].values
    #----
    from timeit import default_timer as timer
    TIME = timer()
    # print(geopt)
    geopt_for_ipral = geopt.sel(time=time_unique, latitude=LAT, longitude=LON).to_dataframe()#['geopt']
    ta_for_ipral = ta.sel(time=time_unique, latitude=LAT, longitude=LON).to_dataframe()#['ta']
    print(f'Time loading {timer()-TIME}')
    #----
    print('-----GETTING PRESSURE AND TEMPERATURE-----')
    lat_opar = np.deg2rad(lat_opar)
    acc_gravity = 9.78032*(1+5.2885e-3*(np.sin(lat_opar))**2 - 5.9e-6*(np.sin(2*lat_opar))**2)
    r0 = 2*acc_gravity/(3.085462e-6 + 2.27e-9*np.cos(2*lat_opar) - 2e-12*np.cos(4*lat_opar))
    g0 = 9.80665
    geopt_for_ipral['geopt_height'] = geopt_for_ipral["geopt"]/g0
    geopt_for_ipral['altitude'] = (geopt_for_ipral['geopt_height']*r0)/(acc_gravity*r0/g0 - geopt_for_ipral['geopt_height'])
    M = 28.966E-3 
    R = 8.314510
    T = (15 + 273.15)
    const = -(M*g0)/(R*T)
    p0 = 101325
    geopt_for_ipral['pression'] = p0*np.exp(const*geopt_for_ipral['altitude'])
    output_era = pd.merge(geopt_for_ipral, ta_for_ipral['ta'], left_index=True, right_index=True) 
    print('variables_from_era --> end')
    return output_era


import scipy.interpolate as spi
def interpolate_atb_mol(lidar_name, opar_file, era):     
    """
    the Input is the output dataframe of simulate_atb_mol function
    """
    print('-----BEFORE INTERPOLATE-----')
    d = xr.open_dataset(opar_file)
    r = d.range.values*1e3 + 2160
    timeOpar = d.time.values
    timeEra = np.unique(era.index.get_level_values(1)) 
    time_tmp = np.array(pd.to_datetime(timeOpar).strftime('%Y-%m-%dT%H:00:00')).astype('datetime64[ns]')
    if len(time_tmp) != len(timeOpar):
        print("Time Error")
        sys.exit(1)
    #------
    columns_names = ['altitude', 'pression', 'ta']
    pression_interp, ta_interp = [[] for _ in range(len(columns_names)-1)] 
    new_index = pd.MultiIndex.from_product([timeOpar, r], names = ['time', 'range'])
    # df_new = pd.DataFrame(index = new_index, columns = era.columns)
    print('-----INTERPOLATE ATTENUATED BACKSCATTERING FROM ERA5-----')
    for t1 in time_tmp:
        a = era.loc[pd.IndexSlice[:, t1], columns_names]
        f9 = spi.interp1d(a['altitude'], a['pression'], kind = 'linear', bounds_error=False, fill_value="extrapolate")
        f10 = spi.interp1d(a['altitude'], a['ta'], kind = 'linear', bounds_error=False, fill_value="extrapolate")
        pression_interp, ta_interp = np.append(pression_interp, np.array(f9(r))), np.append(ta_interp, np.array(f10(r)))

    new_df = pd.DataFrame(index = new_index, data = np.array([pression_interp, ta_interp]).T, columns = columns_names[1:]) 
    
    print(Path("/homedata/nmpnguyen/OPAR/Processed/",lidar_name.upper(),opar_file.name.split('.')[0]+"_simul.pkl"))
    new_df.to_pickle(Path("/homedata/nmpnguyen/OPAR/Processed/",lidar_name.upper(),opar_file.name.split('.')[0]+"_simul.pkl"))
    print('interpolate_atb_mol --> end')
    return new_df


from argparse import Namespace, ArgumentParser
# def main():
#     parser = ArgumentParser()
#     parser.add_argument("--folder", "-f", type=str, help="Main folder of lidar data", required=True)
#     parser.add_argument("--date", "-d", type=dt.date.fromisoformat(), help = "YYYY-MM-DD daily file")
#     parser.add_argument("--wavelength", "-w", type=str, help="Name of lidar on upper character", required=True)
#     parser.add_argument("--output", "-o", type=str, help="Output folder path", required=True)
#     opts = parser.parse_args()
#     print(opts)
#     era = pd.read_pickle("/homedata/nmpnguyen/OPAR/Processed/pression_tempe_backscatter_2019_v2.pkl")
#     # read firstly data under 5km 
#     era = era.sort_index(ascending= True)
#     if opts.wavelength==355:    
#         lidar_name = "LI1200"
#         opar_folder = Path(opts.folder)/Path("LI1200.daily")  
#     else: 
#         lidar_name = "LIO3T"
#         opar_folder = Path(opts.folder)/Path("LIO3T.daily")

#     if opts.date is None:
#         data_file = sorted(opar_folder.glob("2019*.nc4"))
#         for path_raw in data_file:
#             simulate_betamol(path_raw, lidar_name, era, Path(opts.output))
#     else:
#         data_file = opar_folder / Path(str(date)+".nc4")
#         simulate_betamol(data_file, lidar_name, era, Path(opts.output))

def main():
    ### Ouvrir le fichier des jours sélectionés
#     with open('/home/nmpnguyen/Codes/ClearSkyLIO3Tlist', 'r') as f:
#         all_data = [line.strip() for line in f.readlines()]

#     metadata_line = all_data[:4]
#     listdays = all_data[4:]
#     w = int(all_data[3].split(': ')[1])
#     for l in listdays:
#         print(Path('/home/nmpnguyen/OPAR/LIO3T.daily/', l+'.nc4'))
#         oparpath = Path('/home/nmpnguyen/OPAR/LIO3T.daily/', l+'.nc4')
#         outputERA = variables_from_era(oparpath)
#         output = interpolate_atb_mol("lio3t", oparpath, outputERA)
    
#     era = era.sort_index(ascending= True)
    parser = ArgumentParser()
    parser.add_argument("--wavelength", "-w", type=int, help="Name of lidar on upper character", required=True)
    opts = parser.parse_args()
    print(opts)
    if opts.wavelength==355:    
        lidar_name = "LI1200"
        opar_folder = Path("/homedata/noel/OPAR/LI1200.daily")  
    else: 
        lidar_name = "LIO3T"
        opar_folder = Path("/homedata/noel/OPAR/LIO3T.daily")
    
    data_file = sorted(opar_folder.glob("2018*.nc4"))
    for oparpath in data_file:
        print(oparpath)
        outputERA = variables_from_era(oparpath)
        output = interpolate_atb_mol(lidar_name.lower(), oparpath, outputERA)
    
if __name__ == '__main__':
    main()