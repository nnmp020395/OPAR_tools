# Le Script est pour interpoler les données manquantes dans la simulation, au long de l'altitude des données opar
# transformer le time opar à meme format avec le temps era5

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
import matplotlib.dates as plt_dates 
# from matplotlib.backends.qt_compat import QtGui
import matplotlib.pyplot as plt
from argparse import Namespace, ArgumentParser

#read li1200
import glob



#plot era5 temperature
# eraTempe = era['ta'].unstack(level=1)
# f, ax = plt.subplots()
# plot = ax.pcolormesh(eraTempe.index.get_level_values(0), eraTempe.columns, eraTempe.T, shading = "nearest")
# ax.xaxis.set_major_formatter(plt_dates.DateFormatter('%Y-%m-%d'))
# plt.setp(ax.get_xticklabels(), rotation = 30)
# ax.set(xlabel = "Time", ylabel = "Level", title = "Temperature (K) ERA5 data")
# colorbar = plt.colorbar(plot, ax=ax) 
# plt.tight_layout()
# plt.savefig("tempe_era5_year_2019.png")


# for m, li1200_path in zip(range(65, 75),files[65:75]):
def simulate_betamol(li1200_path, lidar_name, era):
     #li1200_path = "/home/nmpnguyen/OPAR/LI1200.daily/2019-01-21.nc4"
    print(li1200_path)
    li1200 = nc4.Dataset(li1200_path, 'r')
    day = li1200_path.split("/")[5].split(".")[0]
    rangeLi = li1200.variables['range'][:]
    altLi = rangeLi + 2.1
    zli1200 = altLi*1000
    # read firtly data under 5km
    rSelectLi = np.array(rangeLi[np.where(altLi < 25)]) #en km
    zSelectLi = np.array(zli1200[np.where(altLi < 25)]) #em m
    # liSelect = pd.DataFrame(li1200.variables['signal'][:,np.where(zli1200 < 25000)[0],5])
    # liSelect = pd.DataFrame(li1200.variables['signal'][:,np.where(zli1200 < 25000)[0],6] + li1200.variables['signal'][:,np.where(zli1200 < 25000)[0],7])
    eraSelect = era.loc[day+" 00:00:00":day+" 18:00:00"]#.loc[era['altitude'] < 5000]
    #----------
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
    columns_names = ['altitude', 'pression', 'ta', 'beta355', 'beta532', 'alpha355', 'alpha532', 'tau355', 'tau532', 'beta355mol', 'beta532mol']
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


#read li1200


# for m, li1200_path in zip(range(0, 20),files[:20]):
def simulate_betamol2(li1200_path):
    print(m, li1200_path)   # li1200_path = "/home/nmpnguyen/OPAR/LIO3T.daily/2019-01-21.nc4"
    li1200 = nc4.Dataset(li1200_path, 'r')
    day = li1200_path.split("/")[5].split(".")[0]
    rangeLi = li1200.variables['range'][:]
    altLi = rangeLi + 2.1
    zli1200 = altLi*1000
    # read firtly data under 5km
    rSelectLi = np.array(rangeLi[np.where(altLi < 25)])
    zSelectLi = np.array(zli1200[np.where(altLi < 25)])
    # liSelect = pd.DataFrame(li1200.variables['signal'][:,np.where(zli1200 < 25000)[0],5])
    # liSelect = pd.DataFrame(li1200.variables['signal'][:,np.where(zli1200 < 25000)[0],6] + li1200.variables['signal'][:,np.where(zli1200 < 25000)[0],7])
    eraSelect = era.loc[day+" 00:00:00":day+" 18:00:00"]#.loc[era['altitude'] < 5000]
    #----------
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
    columns_names = ['altitude', 'pression', 'ta', 'beta355', 'beta532', 'alpha355', 'alpha532', 'tau355', 'tau532', 'beta355mol', 'beta532mol']
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
    new_df.to_pickle("/homedata/nmpnguyen/OPAR/Processed/LIO3T/"+day+"_simul.pkl")


def main():
    parser = ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, help="Main folder of lidar data")
    parser.add_argument("--day", "-d", type=str, help = "YYYY-MM-DD daily file")
    parser.add_argument("--lidar_name", "-l", type=str, help="Name of lidar on upper character", required=True)
    opts = parser.parse_args()
    args = Namespace()
    print(opts)
    #read ERA5
    era = pd.read_pickle("/homedata/nmpnguyen/OPAR/Processed/pression_tempe_backscatter_2019_v2.pkl")
    # read firstly data under 5km 
    era = era.sort_index(ascending= True)
    if opts.day is None:
        folder_path = "/home/nmpnguyen/OPAR/"+opts.lidar_name.upper()+".daily/2019*"
        files = [file for file in glob.glob(folder_path)]
        for m, li1200_path in zip(range(0, len(files)),files):
            print(m, li1200_path)
            simulate_betamol(li1200_path, opts.lidar_name, era)
    else:
        li1200_path = "/home/nmpnguyen/OPAR/"+opts.lidar_name.upper()+".daily/"+opts.day+".nc4"
        simulate_betamol(li1200_path, opts.lidar_name, era)


if __name__ == '__main__':
    main()