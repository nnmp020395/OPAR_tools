import netCDF4 as nc4 
import numpy as np
import xarray as xr
import pandas as pd
import datetime 
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as plt_dates 
# from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid



import glob

files = []
for file in glob.glob("/home/nmpnguyen/OPAR/LI1200.daily/*"):
    files.append(file)
for i in range(0, len(files)):
    print(files[i], i)

def get_path_out(main_path, time_index, lidar_name, channel):
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
    path_file_out = main_path + '/' + lidar_name + '/' + year + "/" + month + "/" + day + "/" + file_name_out 
    return path_file_out

signal_final = pd.DataFrame()
for i in range(52, len(files)) :
    file = files[i]
    print(file)
    df = nc4.Dataset(file, 'r')
    time = df.variables['time'][:]
    time_convert = nc4.num2date(df.variables['time'][:], df.variables['time'].units, df.variables['time'].calendar,
         only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    signal_verylow = df.variables['signal'][:,:,5]
    z = df.variables['range'][:]
    signal_verylow1 = signal_verylow[:, 0:z[np.where(z < 20)].shape[0]]
    time_convert = time_convert.reshape((time_convert.shape[0],1))
    # signal = pd.DataFrame(data = np.concatenate((time_convert, signal_verylow1), axis = 1))
    # signal1 = signal.set_index(0).astype(float).groupby(pd.Grouper(freq = '2min')).mean()#aggregate hourly
    # signal_final = pd.concat([signal_final, signal1])
    signal_final = signal_verylow1
    # make a plot
    # xx_convert, zz = np.meshgrid(signal_final.index.to_pydatetime(),z[np.where(z<20)])
    xx_convert, zz = np.meshgrid(time_convert,z[np.where(z<20)])
    title = file.split("/") 
    fig, ax = plt.subplots() 
    p = ax.pcolor(xx_convert, zz, np.log10(signal_final.T))#, vmin=0, vmax=2500)#, cmap = 'RdBu', vmax = A.max())
    xFmt = plt_dates.DateFormatter('%Y-%m-%d %H:%M')
    ax.xaxis.set_major_formatter(xFmt)
    fig.autofmt_xdate()
    plt.setp(ax.get_xticklabels(), rotation = 30)
    plt.xlabel("datetime")
    plt.ylabel("altitude (km)")
    plt.title("chanel = "+ df.variables['channel'][5]+"\n"+title[5].split(".")[0])
    colorbar = plt.colorbar(p, ax=ax)
    colorbar.ax.set_ylabel('log10(signal)')
    plt.tight_layout()
    plt.savefig("/homedata/nmpnguyen/OPAR/Fig/LI1200/"+title[5].split(".")[0]+"_"+df.variables['channel'][5]+"pcolor.png")

for j in range(0, signal1.shape[0]):
        plt.close()
        fig, (ax1, ax2) = plt.subplots(1,2) 
        title = file.split("/") 
        channel_name = "chanel = "+ df.variables['channel'][5]+"\n"+title[5].split(".")[0]+" "+str(signal1.index[j]).split(" ")[1]
        print(channel_name)
        fig.suptitle(channel_name)
        ax1.plot((signal1.iloc[j,:]),z[np.where(z<20)])
        ax1.set(xlabel = "(signal)", ylabel = "altitude (km)")
        ax1.set_ylim(0.0, 20.0)
        ax2.plot(np.log10(signal1.iloc[j,:]),z[np.where(z<20)])
        ax2.set(xlabel = "log10(signal)", ylabel = "altitude (km)")
        ax2.set_ylim(0.0, 20.0)
        fig.savefig(get_path_out("/homedata/nmpnguyen/OPAR/Fig", signal1.index[j], "LI1200", df.variables['channel'][5]))
        # fig.savefig("/home/nmpnguyen/test_fig/LI1200/"+title[5].split(".")[0]+" "+str(signal1.index[j]).split(" ")[1]+""/home/nmpnguyen/test_fig/LI1200/"+title[5].split(".")[0]+" "+str(signal1.index[j]).split(" ")[1]+".png").png")

#--------------------------------------------------

import glob

files = []
for file in glob.glob("/home/nmpnguyen/OPAR/LIO3T.daily/*"):
    files.append(file)

signal_final = pd.DataFrame()
for i in range(0, 1) :
    file = files[i]
    print(file)
    df = nc4.Dataset(file, 'r')
    time = df.variables['time'][:]
    time_convert = nc4.num2date(df.variables['time'][:], df.variables['time'].units, df.variables['time'].calendar)
    signal_verylow = df.variables['signal'][:,:,6:8]
    m = signal_verylow.mean(axis=2)
    z = df.variables['range'][:]
    signal_verylow1 = signal_verylow[:, 0:z[np.where(z < 20)].shape[0],:]
    time_convert = time_convert.reshape((time_convert.shape[0],1))
    channel_LIO3T = df.variables['channel'][6:8]
    fig, axs = plt.subplots(2,2, sharex=True, sharey=True) 
    title = file.split("/")   
    # make multiplots in 1 figure
    for ax, c, j in zip(axs.ravel(), channel_LIO3T, range(0,signal_verylow.shape[2])):
        signal = pd.DataFrame(data = np.concatenate((time_convert, signal_verylow1[:,:,j]), axis = 1))
        signal1 = signal.set_index(0).astype(float).groupby(pd.Grouper(freq = 'H')).mean()    
        signal_final = signal1
        xx_convert, zz = np.meshgrid(signal_final.index.to_pydatetime(),z[np.where(z<20)])
        p = ax.pcolor(xx_convert, zz, np.log10(signal_final.T))#, vmin=0, vmax=2500)#, cmap = 'RdBu', vmax = A.max())
        xFmt = plt_dates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(xFmt)
        fig.autofmt_xdate()
        plt.setp(ax.get_xticklabels(), rotation = 30)
        ax.set(xlabel = "datetime", ylabel= "altitude (km)")
        ax.set_title("channel = %s" % c)
        colorbar = plt.colorbar(p, ax=ax)
        colorbar.ax.set_ylabel('log10(signal)')   
    fig.suptitle("profil of %s" % title[5])   
    plt.tight_layout(rect=[0,0.03,1,0.95])     
    plt.savefig("/home/nmpnguyen/test_fig/"+"LIO3T "+title[5]+".png")

# plot specifiquement mean of 532.p and 532.s 
for i in range(0, 1) : 
    file = files[i]
    print(file)
    title = file.split("/")   
    df = nc4.Dataset(file, 'r')
    time = df.variables['time'][:]
    time_convert = nc4.num2date(df.variables['time'][:], df.variables['time'].units, df.variables['time'].calendar)
    signal_ps, signal_o, z = df.variables['signal'][:,:,6:8], df.variables['signal'][:,:,5], df.variables['range'][:]
    m_ps, m_o = signal_ps[:, 0:z[np.where(z < 20)].shape[0], :], signal_o[:, 0:z[np.where(z < 20)].shape[0]]
    m_ps = m_ps.mean(axis=2)
    time_convert = time_convert.reshape((time_convert.shape[0],1))
    xx_convert, zz = np.meshgrid(time_convert,z[np.where(z<20)])
    fig, (ax1, ax2) = plt.subplots(1,2, sharex = True, sharey = True)
    signal_final_ps = pd.DataFrame(data = np.concatenate((time_convert, m_ps), axis = 1))
    signal_final_o = pd.DataFrame(data = np.concatenate((time_convert, m_o), axis = 1))
    plot_ps = ax1.pcolor(xx_convert, zz, np.log10(m_ps.T))
    plot_o = ax2.pcolor(xx_convert, zz, np.log10(m_o.T))
    xFmt = plt_dates.DateFormatter('%H:%M')
    ax1.xaxis.set_major_formatter(xFmt)
    ax2.xaxis.set_major_formatter(xFmt)
    fig.autofmt_xdate()
    plt.setp(ax1.get_xticklabels(), rotation = 30)
    plt.setp(ax2.get_xticklabels(), rotation = 30)
    ax1.set(xlabel = "datetime", ylabel= "altitude (km)")
    ax2.set(xlabel = "datetime", ylabel= "altitude (km)")
    ax1.set_title("average 00532.p and 00532.s")
    ax2.set_title("00532.o")
    colorbar1, colorbar2 = plt.colorbar(plot_ps, ax=ax1), plt.colorbar(plot_o, ax=ax2)
    colorbar1.set_label("log10(signal")
    colorbar2.set_label("log10(signal")
    fig.suptitle("profil of %s" % title[5])  
    plt.tight_layout(rect=[0,0.03,1,0.95]) 
    plt.savefig("/home/nmpnguyen/test_fig/"+"LIO3T_channels(1)"+title[5]+".png")

for i in range(0, 1):
    file = files[i]
    title = file.split("/") 
    print(file)
    df = nc4.Dataset(file, 'r')
    time, z, channel = df.variables['time'][:], df.variables['range'][:], df.variables['channel'][:]
    signal_o, signal_p, signal_s = df.variables['signal'][:,:,5], df.variables['signal'][:,:,6], df.variables['signal'][:,:,7]


def plot_altitude_time_signal(altitude, time, signal, units_time, calendar, channel, ax=None):
    time_convert = nc4.num2date(time, units_time, calendar)
    time_convert = time_convert.reshape((time_convert.shape[0],1))
    xx_convert, zz = np.meshgrid(time_convert,altitude)
    if channel == "00532.o":
        channel = channel+"-none"
    elif channel == "00532.p":
        channel = channel+"-parallel"
    elif channel == "00532.s":
        channel = channel+"-crossed"
    # fig, ax = plt.subplots(1,1) #, title = channel
    if ax is None:
        ax = plt.gca()
    plot = ax.pcolor(xx_convert, zz, np.log10(signal.T))
    xFmt = plt_dates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(xFmt)
    fig.autofmt_xdate()
    ax.set(xlabel = "Time", ylabel = "Altitude (km)", title = channel)
    colorbar = plt.colorbar(plot, ax=ax)
    colorbar.set_label("log10(signal)")
    return ax

altitude = z[np.where(z<20)]
Fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey = True, sharex = False)
signal = signal_o[:, 0:altitude.shape[0]]
plot_altitude_time_signal(altitude, df.variables['time'][:], signal, df.variables['time'].units, df.variables['time'].calendar, channel[5], ax = ax1)
signal = signal_p[:, 0:altitude.shape[0]]
plot_altitude_time_signal(altitude, df.variables['time'][:], signal, df.variables['time'].units, df.variables['time'].calendar, channel[6], ax = ax2)
signal = signal_s[:, 0:altitude.shape[0]]
plot_altitude_time_signal(altitude, df.variables['time'][:], signal, df.variables['time'].units, df.variables['time'].calendar, channel[7], ax = ax3)
Fig.suptitle("profil of %s" % title[5])  
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.savefig("/home/nmpnguyen/test_fig/test.png")
