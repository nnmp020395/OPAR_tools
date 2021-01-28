import netCDF4 as nc4 
import numpy as np
# import xarray as xr
import pandas as pd
import datetime 
from datetime import timedelta
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as plt_dates 
import glob, os
# from matplotlib.backends.qt_compat import QtGui

def get_path_out(main_path, time_index, lidar_name, channel, arranging = False):
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


def get_signal_significatif(signal):
    array_nozero = np.where(signal)
    if len(array_nozero[0])/(signal.shape[0]*signal.shape[1]) > 0.1:
        return signal
    else:
        return None 


def time_from_opar_raw(path):
    data = nc4.Dataset(path, 'r')
    time, calendar, units_time = data.variables['time'][:], data.variables['time'].calendar, data.variables['time'].units
    timeLidar = pd.to_datetime(nc4.num2date(time, units_time, calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True))
    # timeLidar = np.array(timeLidar.strftime("%Y-%m-%d %H:%M")).astype("datetime64[s]")
    return timeLidar


def plot_altitude_time_signal(altitude, time, signal, units_time, calendar, channel, ax=None):
    time1 = nc4.num2date(time, units_time, calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    time_convert = time1.reshape((time1.shape[0],1))
    xx_convert, zz = np.meshgrid(time_convert,altitude)
    if channel == "00532.o":
        channel = channel+"-none"
    elif channel == "00532.p":
        channel = channel+"-parallel"
    elif channel == "00532.s":
        channel = channel+"-crossed"
    if ax is None:
        ax = plt.gca()
    # fig, ax = plt.subplots(1,1) #, title = channel
    plot = ax.pcolor(xx_convert, zz, np.log10(signal.T))
    xFmt = plt_dates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(xFmt)
    ax.set(xlabel = "Time", ylabel = "Altitude (km)", title = channel)
    colorbar = plt.colorbar(plot, ax=ax, ticks=range(0,5,1))
    colorbar.set_label("log10(signal)")
    return ax


files = []
for file in glob.glob("/home/nmpnguyen/OPAR/LIO3T.daily/*"):
    files.append(file)


# files = np.delete(files, np.where(np.isin(files, ['2019-02-19.nc4', '2019-11-19.nc4'])))
# for i in range(62, len(files)):
#     file = files[i]
#     print(i, file)
#     df = nc4.Dataset(file, 'r')
#     time2= time_from_opar_raw(file)
#     z2 = df.variables['range'][np.where(df.variables['range'][:]+2.1 < 25)]
#     signal2 = pd.DataFrame((df.variables['signal'][:,:z2.shape[0],6] + df.variables['signal'][:,:z2.shape[0],7]))
#     for t in time2:
#         t = pd.to_datetime(t)
#         Fig, (ax, ax2)= plt.subplots(1, 2, sharey=True)
#         ax.plot(signal2.iloc[np.where(time2 == t)[0][0], :], z2)
#         ax.set(xlabel = "(signal)", ylabel = "altitude (km)")
#         ax2.semilogx((signal2.iloc[np.where(time2 == t)[0][0], :]), z2)
#         ax2.set(xlabel = "log10(signal)")
#         Fig.suptitle("channel = p+s \n"+str(t))
#         plt.savefig(get_path_out(main_path = "/homedata/nmpnguyen/OPAR/Fig", time_index = t, lidar_name = "LIO3T", channel = "profil2channels", arranging = "day"))


from matplotlib import colors
for i in range(0, len(files)):
    file = files[i]
    print(i, file)
    df = nc4.Dataset(file, 'r')    
    signal_sum = df.variables['signal'][:,:,6] + df.variables['signal'][:,:,7]     
    z = df.variables['range'][:]+2.1
    altitude = z[np.where(z<25)]
    signal = pd.DataFrame(signal_sum[:,:altitude.shape[0]])
    time2= time_from_opar_raw(file)
    if get_signal_significatif(signal) is None:
        print("Signal is None")
    else:
        Fig, ax = plt.subplots()
        plot = ax.pcolormesh(time2, altitude, signal.T, shading="nearest", norm = colors.LogNorm())
        ax.xaxis.set_major_formatter(plt_dates.DateFormatter('%H:%M'))
        ax.set(xlabel = "Time", ylabel = "Altitude (km)", title = "00532.p+00532.s")
        colorbar = plt.colorbar(plot, ax=ax)
        colorbar.ax.set_ylabel("(signal)")
        Fig.autofmt_xdate()
        title = file.split("/") 
        Fig.suptitle("profil of %s" % title[5])
        plt.tight_layout()
        time_index = pd.to_datetime(title[5].split(".")[0])
        plt.savefig(get_path_out(main_path = "/homedata/nmpnguyen/OPAR/Fig", time_index = time_index, lidar_name = "LIO3T", channel = "2channels", arranging = "month"))
        plt.clf()
        plt.close()


# 1: plot brut profil vartical
# 2: plot brut pcolor
# 3: arrange : pcolor by months, vertical profile by day 