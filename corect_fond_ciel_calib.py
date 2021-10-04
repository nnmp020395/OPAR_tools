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

def get_path_out(main_path, time_index, lidar_name, channel, arranging = False):
    time_index = pd.to_datetime(time_index)
    year = str(time_index.year)
    month = time_index.month
    day = time_index.day
    if month < 10 :
        month = '0' + str(month)
    else:
        month = str(month)
    if day < 10:
        day = '0' + str(day)
    else:
        day = str(day)   
    file_name_out = str(time_index) + "_" + channel + ".png"
    if  arranging == "day":
        path_file_out = main_path + '/' + lidar_name + '/' + year + "/" + month + "/" + day + "/" + file_name_out 
    elif arranging == "month":
        path_file_out = main_path + '/' + lidar_name + '/' + year + "/" + month + "/" + file_name_out
    else:
        path_file_out = main_path + '/' + lidar_name + '/' + year + "/" + file_name_out
    return path_file_out


path_raw = glob.glob("/home/nmpnguyen/OPAR/LI1200.daily/*.nc4")
# for path in path_raw[:1]:
path = "/home/nmpnguyen/OPAR/LIO3T.daily/2019-06-17.nc4"
print(path)
dt = nc4.Dataset(path, 'r')
altLi = np.array(dt.variables['range'][:][np.where(dt.variables['range'][:]<=100)])
rangeLi = altLi - 2.1
rangeLi2 = (rangeLi*1000)**2
time = time_from_opar_raw(path)
# signal = np.array(dt.variables['signal'][:,np.where(altLi<25)[0],5]) #LI1200
#LIO3T
signal = np.array(dt.variables['signal'][:,np.where(altLi<25)[0],6]) + np.array(dt.variables['signal'][:,np.where(altLi<25)[0],7]) 
fc = np.array(dt.variables['signal'][:,np.where(altLi>80)[0],5])
print("#---------Average Fond ciel")
mean_fc = fc.mean(axis =1).reshape(-1, 1)
signal_nofc = pd.DataFrame(signal - mean_fc)
signal_nofc2 = signal_nofc*rangeLi2[np.where(altLi<25)]
for i in range(0,2):#range(0, signal_nofc.shape[0]):
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(1,4, sharey=True, figsize=(12, 5))
    ax[0].plot(signal[i,:], altLi[np.where(altLi<25)])
    ax[0].set(xlabel="Signal raw", ylabel="Altitude (km)")
    ax[0].vlines(mean_fc,ymin=altLi[0], ymax=altLi[np.where(altLi<25)][-1], linestyles="--", label="sky background")
    ax[0].legend()
    ax[1].plot(signal_nofc.iloc[i,:], altLi[np.where(altLi<25)])        
    ax[1].set(xlabel="Signal raw", ylabel="Altitude (km)")
    ax[1].set_title("no Sky background", loc='right')
    ax[2].plot(rangeLi2[np.where(altLi<25)], altLi[np.where(altLi<25)], label="range^2")
    ax[2].plot((altLi[np.where(altLi<25)]*1000)**2, altLi[np.where(altLi<25)], label="alt^2")
    ax[2].set(xlabel="Range(m)^2 ", ylabel="Altitude (km)")
    ax[2].legend()
    ax[3].plot(signal_nofc.iloc[i,:]*rangeLi2[np.where(altLi<25)], altLi[np.where(altLi<25)])
    ax[3].set(xlabel="Signal raw*Range(m)^2 ", ylabel="Altitude (km)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # fig.suptitle("LI1200 Low Channel"+str(time[i])) #LI12OO
    fig.suptitle("LIO3T p+s Channel"+str(time[i])) #LIO3T
    # plt.savefig(get_path_out(main_path="/homedata/nmpnguyen/OPAR/Fig", time_index=time[i], lidar_name="LI1200", channel="AllRaw", arranging = "day"))
    plt.savefig(get_path_out(main_path="/homedata/nmpnguyen/OPAR/Fig", time_index=time[i], lidar_name="LIO3T", channel="AllRaw", arranging = "day"))
    


print("#---------Check Clear sky")
alt_cc = altLi[np.where((altLi<=4.9)&(altLi>=4.8))[0]]*1000
ccr2 = pd.DataFrame(signal_nofc2.loc[:,np.where((altLi<=4.9)&(altLi>=4.8))[0]])#, index=time)
ccr2.columns = alt_cc; ccr2.index = time
    #---------Set Clear sky simulated
path_simu = "/homedata/nmpnguyen/OPAR/Processed/LIO3T/"+path.split("/")[5].split(".")[0]+"_simul.pkl"
beta355mol = pd.read_pickle(path_simu)['beta532mol'].unstack(level=1)
cc_simu = beta355mol[alt_cc]
const = np.array(ccr2.div(cc_simu, axis=1).mean(axis=1))
cc_calib = ccr2.div(const.reshape(const.shape[0],1), axis=0)
beta355_calib = signal_nofc2.div(const.reshape(const.shape[0],1), axis=0)
beta355_calib.index=time; beta355_calib.columns=altLi[np.where(altLi<25)]*1000
SR355 = beta355_calib.div(beta355mol)
for i in range(2,10):#beta355_calib.shape[0]):
    ccr2 = pd.DataFrame(data = signal_nofc.loc[i,np.where((altLi<=4.9)&(altLi>=4.8))[0]])
    ccr2.index = alt_cc#, index=time)    
    cc_simu = beta355mol[alt_cc].iloc[i]
    const = np.array(ccr2.div(cc_simu, axis=0).mean(axis=0))
    cc_calib = ccr2/const#.div(const.reshape(const.shape[0],1), axis=0)
    beta355_calib = signal_nofc2.loc[i]/const#.div(const.reshape(const.shape[0],1), axis=0)
    beta355_calib.index = altLi[np.where(altLi<25)]*1000
    SR355 = beta355_calib.div(beta355mol.iloc[i])
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(1,4, sharey=True, figsize=(12, 5))
    ax[0].semilogx(signal_nofc2.iloc[i,:], altLi[np.where(altLi<25)])
    ax[0].set(xlabel="Signal*Range(m)^2 ", ylabel="Altitude (km)")
    ax[0].hlines(4.9, xmin=signal_nofc2.iloc[i].min(), xmax=signal_nofc2.iloc[i].max(), color="green", linestyles="-.")
    ax[0].hlines(4.8, xmin=signal_nofc2.iloc[i].min(), xmax=signal_nofc2.iloc[i].max(), color="green", linestyles="-.")
    ax[1].semilogx(beta355mol.iloc[i], altLi[np.where(altLi<25)])
    ax[1].set(xlabel="ATB mol", ylabel="Altitude (km)")
    ax[1].set_title("Coef.calib= %1.4e" %const)
    ax[1].hlines(4.9, xmin=beta355mol.iloc[i].min(), xmax=beta355mol.iloc[i].max(), color="green", linestyles="-.")
    ax[1].hlines(4.8, xmin=beta355mol.iloc[i].min(), xmax=beta355mol.iloc[i].max(), color="green", linestyles="-.")
    ax[2].semilogx(beta355_calib, altLi[np.where(altLi<25)], label="calibrated measured")
    ax[2].semilogx(beta355mol.iloc[i], altLi[np.where(altLi<25)], "--", label="simulated molecular")
    ax[2].hlines(4.9, xmin=beta355_calib.min(), xmax=beta355_calib.max(), color="green", linestyles="-.")
    ax[2].hlines(4.8, xmin=beta355_calib.min(), xmax=beta355_calib.max(), color="green", linestyles="-.")
    ax[2].set(xlabel="ATB", ylabel="Altitude (km)")
    ax[2].legend()    
    ax[3].plot(SR355, altLi[np.where(altLi<25)], label="calibrated measured SR")
    ax[3].vlines(1, ymin=altLi[0], ymax=altLi[np.where(altLi<25)][-1], linestyles="--", color="red",label="theoretical SR \nclear sky")
    ax[3].hlines(4.9, xmin=SR355.min(), xmax=SR355.max(), color="green", linestyles="-.")
    ax[3].hlines(4.8, xmin=SR355.min(), xmax=SR355.max(), color="green", linestyles="-.")
    ax[3].set(xlabel="SR", ylabel="Altitude (km)")
    ax[3].legend()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # fig.suptitle("Calibration LI1200 Low Channel"+str(time[i]))
    # plt.savefig(get_path_out(main_path="/homedata/nmpnguyen/OPAR/Fig", time_index=time[i], lidar_name="LI1200", channel="AllCalib", arranging = "day"))
    fig.suptitle("Calibration LIO3T p+s Channel"+str(time[i]))
    plt.savefig(get_path_out(main_path="/homedata/nmpnguyen/OPAR/Fig", time_index=time[i], lidar_name="LIO3T", channel="AllCalib", arranging = "day"))

