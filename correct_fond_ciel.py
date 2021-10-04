# Corriger fond ciel pour les voies 355 et 532 en enlever la différence de signal avec la moyenne entre 80 et 100km 
# attention: range = altitude - 2.1km 

# import xarray as xr 
import numpy as np 
import pandas as pd 
import netCDF4 as nc4
import glob, os
import scipy.interpolate as spi
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg') 
import matplotlib.dates as plt_dates 
# from matplotlib.backends.qt_compat import QtGui
import matplotlib.pyplot as plt
import glob, os

def time_from_opar_raw(path):
    data = nc4.Dataset(path, 'r')
    time, calendar, units_time = data.variables['time'][:], data.variables['time'].calendar, data.variables['time'].units
    timeLidar = pd.to_datetime(nc4.num2date(time, units_time, calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True))
    # timeLidar = np.array(timeLidar.strftime("%Y-%m-%d %H:%M")).astype("datetime64[s]")
    return timeLidar

path_raw = glob.glob("/home/nmpnguyen/OPAR/LI1200.daily/*.nc4")
for path in path_raw:
    print(path)
    dt = nc4.Dataset(path, 'r')
    rangeLi = dt.variables['range'][:][np.where(dt.variables['range'][:]<=100)]
    altLi = rangeLi + 2.1
    time = time_from_opar_raw(path)
    signal = dt.variables['signal'][:,:rangeLi.shape[0],5]
    fc = dt.variables['signal'][:,np.where(rangeLi>80)[0],5]
    #---------Average Fond ciel
    mean_fc = fc.mean(axis =1).reshape(-1, 1)
    signal_nofc = signal - mean_fc
    #---------Check Clear sky
    cc = pd.DataFrame(signal_nofc[:,np.where(rangeLi<6.2)[0]])
    ccr2 = cc*(rangeLi[np.where(rangeLi<6.2)[0]]**2)#cc.mul(rangeLi**2, axis=1)
    mean_ccr2 = ccr2.mean(axis=1)
    print(mean_ccr2)    
    fig, ax=plt.subplots()
    ax.set(xlabel="signal", ylabel="altitude", title="clear sky x r²")
    for i in range(0, ccr2.shape[0]):
        ax.semilogx(ccr2.iloc[i,:], rangeLi[np.where(rangeLi<6.2)[0]])
    plt.savefig("/home/nmpnguyen/li1200_LowAlt"+path.split("/")[5]+".png")




