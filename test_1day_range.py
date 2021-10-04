import numpy as np 
import pandas as pd 
import netCDF4 as nc4
from netCDF4 import Dataset
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
from matplotlib import colors

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
        path_file_out = main_path + '/' + lidar_name.upper() + '/' + year + "/" + month + "/" + day + "/" + file_name_out 
    elif arranging == "month":
        path_file_out = main_path + '/' + lidar_name.upper() + '/' + year + "/" + month + "/" + file_name_out
    else:
        path_file_out = main_path + '/' + lidar_name.upper() + '/' + year + "/" + file_name_out
    return path_file_out


def plot_355_raw(li1200):
    dt1200 = nc4.Dataset(li1200, 'r')
    t1200 = time_from_opar_raw(li1200)
    alti_li1200 = dt1200.variables['range'][:][np.where(dt1200.variables['range'][:]<=100)]
    range_li1200 = alti_li1200 - 2.1
    signal_li1200 = dt1200.variables['signal'][:,:alti_li1200.shape[0],5]
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots(1,2, sharey = True)
    p0=ax[0].pcolormesh(t1200, alti_li1200, signal_li1200.T, label ="alti", shading = "nearest", norm = colors.LogNorm())
    p1=ax[1].pcolormesh(t1200, range_li1200, signal_li1200.T, label = "range", shading = "nearest", norm = colors.LogNorm())
    ax[0].set_ylim(0.0, 25.0)
    ax[1].set_ylim(0.0, 25.0)
    ax[0].legend()
    ax[1].legend()
    plt.setp(ax[0].get_xticklabels(), rotation = 90)
    colorbar0 = plt.colorbar(p0, ax=ax[0])
    plt.setp(ax[1].get_xticklabels(), rotation = 90)
    colorbar1 = plt.colorbar(p1, ax=ax[1])
    fig.suptitle(li1200)
    plt.savefig(get_path_out(main_path="/homedata/nmpnguyen/OPAR/Fig/", time_index=t1200[0], lidar_name="LI1200", channel="LI1200_raw1", arranging="day"))
    # plt.savefig("/homedata/nmpnguyen/OPAR/Fig/"+fn"_LI1200_raw1.png")

def plot_355_raw(lio3t):
    dtlio3t = nc4.Dataset(lio3t, 'r')
    tlio3t = time_from_opar_raw(lio3t)
    alti_lio3t = dtlio3t.variables['range'][:][np.where(dtlio3t.variables['range'][:]<=100)]
    range_lio3t = alti_lio3t - 2.1
    signal_lio3t = dtlio3t.variables['signal'][:,:alti_lio3t.shape[0],6]+dtlio3t.variables['signal'][:,:alti_lio3t.shape[0],7]
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots(1,2, sharey = True)
    p0=ax[0].pcolormesh(tlio3t, alti_lio3t, signal_lio3t.T, label ="alti", shading = "nearest", norm = colors.LogNorm())
    p1=ax[1].pcolormesh(tlio3t, range_lio3t, signal_lio3t.T, label = "range", shading = "nearest", norm = colors.LogNorm())
    ax[0].set_ylim(0.0, 25.0)
    ax[1].set_ylim(0.0, 25.0)
    ax[0].legend()
    ax[1].legend()
    plt.setp(ax[0].get_xticklabels(), rotation = 90)
    colorbar0 = plt.colorbar(p0, ax=ax[0])
    plt.setp(ax[1].get_xticklabels(), rotation = 90)
    colorbar1 = plt.colorbar(p1, ax=ax[1])
    fig.suptitle(lio3t)
    plt.savefig(get_path_out(main_path="/homedata/nmpnguyen/OPAR/Fig/", time_index=tlio3t[0], lidar_name="LIO3T", channel="LI1200_raw1", arranging="day"))
    # plt.savefig("/homedata/nmpnguyen/OPAR/Fig/"+fn"_LIO3T_raw1.png")

def plot_calibrated_betamol1(dataRaw, dataSimul, dataCalib, zLidar, constK, time_plot, lidar_name, channel):
    f, ax = plt.subplots(1,3, sharey = True)
    ax[0].semilogx(dataRaw.loc[time_plot]*(zLidar**2), zLidar/1000)
    ax[1].semilogx(dataSimul.loc[time_plot], zLidar/1000, '--', label = "molecular simulated ATB")
    ax[1].semilogx(dataCalib.loc[time_plot], zLidar/1000, label = "measured calibrated ATB")
    ax[2].plot(dataCalib.loc[time_plot]/dataSimul.loc[time_plot], zLidar/1000, label = "SR")
    ax[2].vlines(1, ymin=zLidar[0], ymax=zLidar[-1]/1000, linestyles="--", color="red",label="theoretical SR \nclear sky")
    ax[0].set(xlabel = "uncalibrated signal x Z²", ylabel = "Altitude")
    ax[1].set(xlabel = "ATB")
    ax[2].set(xlabel = "SR")
    ax[2].set_title("Calib.Coef. = %1.3e" %constK.loc[time_plot], loc='right')
    plt.tight_layout(rect=[0,0.03,1,0.95])
    f.suptitle("Calibration step for "+lidar_name+"-"+channel+"\n"+str(time_plot))
    # plt.savefig("/home/nmpnguyen/"+"calibrated_step"+str(time_plot)+".png")
    plt.savefig(get_path_out(main_path = "/homedata/nmpnguyen/OPAR/Fig", time_index = time_plot, lidar_name=lidar_name.upper(), channel=channel, arranging = "day"))



def calibration(path_raw, path_simu, file_name, lidar_name):            
    def get_signal_significatif(signal):
        array_nozero = np.where(signal)
        if len(array_nozero[0])/(signal.shape[0]*signal.shape[1]) > 0.1:
            return signal
        else:
            return None 
    new_df = pd.read_pickle(path_simu)
    li1200 = nc4.Dataset(path_raw, 'r')
    timeLi1 = time_from_opar_raw(path_raw)
    altLi = li1200.variables['range'][:][np.where(li1200.variables['range'][:]<=100)]
    rangeLi  = altLi - 2.1
    zli1200 = altLi*1000 #np.array(li1200.variables['range'][:]*1000)+2100
    rSelectLi = np.array(rangeLi[np.where(altLi < 25)])
    zSelectLi = np.array(zli1200[np.where(altLi < 25)])
    if lidar_name == "li1200":
        print("step1")
        signal_before = (li1200.variables['signal'][:,:rangeLi.shape[0],4])
        fc = li1200.variables['signal'][:,np.where(rangeLi>80)[0],4]
        mean_fc = fc.mean(axis =1).reshape(-1, 1)
        beta = "beta355mol"
        channel = li1200.variables['channel'][4]
    else:
        print("step1")
        signal_before = (li1200.variables['signal'][:,:rangeLi.shape[0],6] + li1200.variables['signal'][:,:rangeLi.shape[0],7])
        fc = li1200.variables['signal'][:,np.where(rangeLi>80)[0],6]+li1200.variables['signal'][:,np.where(rangeLi>80)[0],7]
        mean_fc = fc.mean(axis =1).reshape(-1, 1)
        beta = "beta532mol"
        channel = "channel p+s"
    if get_signal_significatif(signal_before) is None:
        print("Signal is None")
        signal_new355 = None
        z_cc = None
        betamol355_simu = None
        constK1 = None
    else:
        print("step2")
        signal = signal_before-mean_fc
        i = np.where(altLi > 4)[0][0]
        r_cc = rSelectLi[i:zSelectLi.shape[0]]
        z_cc = zSelectLi[i:zSelectLi.shape[0]]
        # liSelect = pd.DataFrame(li1200.variables['signal'][:,i:zSelectLi.shape[0],5], index = timeLi1)
        liSelect = pd.DataFrame(signal[:,i:rSelectLi.shape[0]], index = timeLi1)
        liSelect_cc = liSelect.iloc[:,:8].mul(r_cc[:8]**2, axis=1).mean(axis=1)
        newdf_cc = new_df.iloc[np.where((new_df.index.get_level_values(1)).isin(z_cc[:8]))]
        betamol355_cc = newdf_cc[beta].unstack(level=1).mean(axis=1) # mean each day 
        # betamol532_cc = np.array(newdf_cc['beta532mol']).reshape((time.shape[0], 10)).mean(axis=1)
        constK1 = pd.DataFrame(liSelect_cc / betamol355_cc, columns = ["constK"]).astype(np.float64)
        print("step3")
        signal_new355 = pd.DataFrame(signal[:,:rSelectLi.shape[0]], index=timeLi1).mul(rSelectLi**2, axis=1).div(constK1['constK'], axis = 0)
        print(signal_new355.shape)
        # signal_new355.index = signal_new355.index.strftime("%Y-%m-%d %H:%M").astype("datetime64[ns]")
        betamol355_simu = new_df.iloc[np.where((new_df.index.get_level_values(1)).isin(zSelectLi))][beta]
        # betamol355_simu.index = betamol355_simu.index.set_levels(betamol355_simu.index.levels[0].strftime("%Y-%m-%d %H:%M").astype("datetime64[ns]"), level=0)
        betamol355_simu = betamol355_simu.unstack(level=1)
        print(betamol355_simu.shape)
        betamol355_simu.columns = range(0, zSelectLi.shape[0], 1)
        print('plot ATB')
        plt.cla()
        plt.clf()
        for time_plot in timeLi1:#time_plot = timeLi1[5]
            plot_calibrated_betamol1(pd.DataFrame(signal[:,:zSelectLi.shape[0]], index=timeLi1), betamol355_simu, signal_new355, zSelectLi, constK1, time_plot, lidar_name, channel="range")        
    li1200.close()
    return signal_new355, zSelectLi, betamol355_simu, constK1#, signal_new532, z_cc2, betamol532_simu, constK2


fn = "2019-01-24"
li1200 = "/home/nmpnguyen/OPAR/LI1200.daily/"+fn+".nc4"
lio3t = "/home/nmpnguyen/OPAR/LIO3T.daily/"+fn+".nc4"


path_simu = "/homedata/nmpnguyen/OPAR/Processed/LI1200/"+fn+"_simul.pkl"
signal_new355, z_cc, betamol355_simu, constK1 = calibration(li1200, path_simu, fn, "li1200")
t355 = pd.to_datetime(signal_new355.index)

path_simu = "/homedata/nmpnguyen/OPAR/Processed/LIO3T/"+fn+"_simul.pkl"
signal_new532, z_cc2, betamol532_simu, constK2 = calibration(lio3t, path_simu, fn, "lio3t") 
t532 = pd.to_datetime(signal_new532.index)

def plot_quicklook_calib(signal_calib, z, time, lidar_name)
from matplotlib import colors
plt.cla()
plt.clf()
fig, ax = plt.subplots(1,2, sharey = True)
p0 = ax[0].pcolormesh(t532, z_cc2/1000, signal_new532.T, shading = "nearest", norm = colors.LogNorm())
colorbar = plt.colorbar(p0, ax=ax[0])
plt.setp(ax[0].get_xticklabels(), rotation = 90)
ax[0].set(title="ATB")
sr532 = (signal_new532.div(betamol532_simu)).astype("float64")
p1 = ax[1].pcolormesh(t532, z_cc2/1000, sr532.T, shading = "nearest", vmin=0, vmax=30)
colorbar = plt.colorbar(p1, ax=ax[1])
plt.setp(ax[1].get_xticklabels(), rotation = 90)
ax[1].set(title="SR")
plt.suptitle("ATB and SR (range) \n"+lio3t)
plt.tight_layout()
plt.savefig("/homedata/nmpnguyen/OPAR/Fig/"+fn+"_LIO3T_calibRange.png")



plt.cla()
plt.clf()
fig, ax = plt.subplots(1,2, sharey = True)
p0 = ax[0].pcolormesh(t355, z_cc/1000, signal_new355.T, shading = "nearest", norm = colors.LogNorm())
colorbar = plt.colorbar(p0, ax=ax[0])
plt.setp(ax[0].get_xticklabels(), rotation = 90)
ax[0].set(title="ATB")
sr355 = (signal_new355.div(betamol355_simu)).astype("float64")
p1 = ax[1].pcolormesh(t355, z_cc/1000, sr355.T, shading = "nearest", vmin=0, vmax=10)
colorbar = plt.colorbar(p1, ax=ax[1])
plt.setp(ax[1].get_xticklabels(), rotation = 90)
ax[1].set(title="SR")
plt.suptitle("ATB and SR (range) \n"+li1200)
plt.tight_layout()
plt.savefig("/homedata/nmpnguyen/OPAR/Fig/"+fn+"_LI1200_calibRange.png")




def test_sr(sr355, sr532, z1, z2):
    r = np.sort(np.intersect1d(z1, z2))
    alt_real = r[np.where((r>=6000)&(r<=20000))]
    sr355_reshape = sr355[np.intersect1d(z1, alt_real, return_indices = True)[1]]
    sr532_reshape = sr532[np.intersect1d(z2, alt_real, return_indices = True)[1]]
    sr355_reshape.drop_duplicates(keep='first', inplace=True)
    sr532_reshape.drop_duplicates(keep='first', inplace=True)
    time1 = sr355_reshape.index.get_level_values(0).strftime("%Y-%m-%d %H:%M").astype("datetime64[ns]")
    time2 = sr532_reshape.index.get_level_values(0).strftime("%Y-%m-%d %H:%M").astype("datetime64[ns]")
    times = np.sort(np.intersect1d(time1, time2))
    sr355_reshape.index = time1
    sr532_reshape.index = time2
    SR355plot = np.asarray(sr355_reshape.loc[times].sort_index(ascending=True).stack(dropna=False))
    SR532plot = np.asarray(sr532_reshape.loc[times].sort_index(ascending=True).stack(dropna=False))
    # for t in times:
    #     plt.close()
    #     ff, ax = plt.subplots()
    #     ax.plot(sr532_reshape.loc[t], alt_real, color = "green", label="SR532")
    #     ax.plot(sr355_reshape.loc[t], alt_real, color = "red", label="SR355")
    #     ax.legend()
    #     ax.set(xlabel = "SR", ylabel="Altitude")
    #     # plt.savefig("/homedata/nmpnguyen/OPAR/Fig/Calibrated_LI1200_LIO3T/"+"SR_profil"+str(t)+".png")
    #     plt.savefig(get_path_out(main_path="/homedata/nmpnguyen/OPAR/Fig/", time_index=t, lidar_name="LI1200", channel="SRProfil", arranging = "day"))
    #     plt.savefig(get_path_out(main_path="/homedata/nmpnguyen/OPAR/Fig/", time_index=t, lidar_name="LIO3T", channel="SRProfil", arranging = "day"))
    # # remove NAN values on these two SR data 
    SR355plot[(SR355plot == np.inf)|(SR355plot == -np.inf)] = np.nan
    SR532plot[(SR532plot == np.inf)|(SR532plot == -np.inf)] = np.nan
    union_nan = np.union1d(np.argwhere(np.isnan(SR355plot)), np.argwhere(np.isnan(SR532plot)))
    SR355plot = np.delete(SR355plot, union_nan)
    SR532plot = np.delete(SR532plot, union_nan)
    return SR355plot, SR532plot

    
SR355plot, SR532plot = test_sr(sr355, sr532, z_cc, z_cc2)
bins = 100
plt.close()
Fig, axs = plt.subplots()
h = axs.hist2d(SR355plot, SR532plot, bins = bins, vmin= 0, vmax = 30)# , norm=colors.LogNorm())# ) 
# h = np.histogram2d(SR355plot, SR532plot, bins=bins)
# plt.imshow(h[0].T, interpolation='nearest')
axs.set(ylabel = "SR (532nm)", xlabel = "SR (355nm)", title = fn+" bins = "+str(bins) + " (range)") #title = pd.to_datetime(times[0]).strftime("%Y-%m-%d")+" bin="+str(bins)
plt.colorbar(h[3], ax=axs)#, norm=colors.NoNorm
axs.set_xlim(0,10)
axs.set_ylim(0,20)
plt.savefig("/homedata/nmpnguyen/OPAR/Fig/"+fn+"_SRrange.png")

ATB355, ATB532 = test_sr(signal_new355, signal_new532, z_cc, z_cc2)
bins = 100
fig, ax = plt.subplots()
h = ax.hist2d(ATB355, ATB532, bins=bins, norm = colors.LogNorm())
ax.set(xlabel="ATB (355nm)", ylabel="ATB (532nm)", title= fn+" bins=" + str(bins)+ " (range)")
plt.colorbar(h[3], ax=ax)
plt.savefig("/homedata/nmpnguyen/OPAR/Fig/"+fn+"_ATBrange.png")