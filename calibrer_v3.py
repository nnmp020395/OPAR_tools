import xarray as xr 
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

import glob 
# li1200_raw_files = glob.glob("/home/nmpnguyen/OPAR/LI1200.daily/2019*")
# li1200_simu_files = glob.glob("/homedata/nmpnguyen/OPAR/Processed/LI1200/2019*.pkl")
# lio3t_raw_files = glob.glob("/home/nmpnguyen/OPAR/LIO3T.daily/2019*")
# lio3t_simu_files = glob.glob("/homedata/nmpnguyen/OPAR/Processed/LIO3T/2019*.pkl")




def calibratiion(path_355_raw, path_355_simu, path_532_raw, path_532_simu):    
    def time_from_opar_raw(path):
        data = nc4.Dataset(path, 'r')
        time, calendar, units_time = data.variables['time'][:], data.variables['time'].calendar, data.variables['time'].units
        timeLidar = pd.to_datetime(nc4.num2date(time, units_time, calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True))
        timeLidar = np.array(timeLidar.strftime("%Y-%m-%d %H:%M")).astype("datetime64[s]")
        return timeLidar
    new_df = pd.read_pickle(path_355_simu)
    li1200 = nc4.Dataset(path_355_raw, 'r')
    # time, calendar, units_time = li1200.variables['time'][:], li1200.variables['time'].calendar, li1200.variables['time'].units
    # timeLi1 = pd.to_datetime(nc4.num2date(time, units_time, calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True))
    timeLi1 = time_from_opar_raw(path_355_raw)
    zli1200 = np.array(li1200.variables['range'][:]*1000)
    zSelectLi = np.array(zli1200[np.where(zli1200 < 20000)])
    signal = pd.DataFrame(li1200.variables['signal'][:,:zSelectLi.shape[0],5])
    # i = signal.where(signal > 0.0).stack().index.values[0][1]
    i = np.where(zSelectLi > 4000)[0][0]
    z_cc = zli1200[i:zSelectLi.shape[0]]
    liSelect = pd.DataFrame(li1200.variables['signal'][:,i:zSelectLi.shape[0],5], index = timeLi1)
    liSelect_cc = liSelect.iloc[:,:10].mul(z_cc[:10]**2, axis=1).mean(axis=1)
    newdf_cc = new_df.iloc[np.where(new_df.index.get_level_values(1).isin(z_cc[:10]))]
    betamol355_cc = np.array(newdf_cc['beta355mol']).reshape((timeLi1.shape[0], z_cc[:10].shape[0])).mean(axis=1) # mean each day 
    # betamol532_cc = np.array(newdf_cc['beta532mol']).reshape((time.shape[0], 10)).mean(axis=1)
    constK = pd.DataFrame(liSelect_cc / betamol355_cc, index=timeLi1, columns = ["constK"]).astype(np.float64)
    signal_new355 = liSelect.div(constK['constK'], axis = 0).mul(z_cc**2, axis=1)
    # signal_new355.index = signal_new355.index.strftime("%Y-%m-%d %H:%M").astype("datetime64[ns]")
    betamol355_simu = new_df.iloc[np.where(new_df.index.get_level_values(1).isin(z_cc))]['beta355mol']
    betamol355_simu.index = betamol355_simu.index.set_levels(betamol355_simu.index.levels[0].strftime("%Y-%m-%d %H:%M").astype("datetime64[ns]"), level=0)
    betamol355_simu = betamol355_simu.unstack(level=1)
    betamol355_simu.columns = range(0, z_cc.shape[0], 1)
    new_df = pd.read_pickle(path_532_simu)
    lio3t = nc4.Dataset(path_532_raw, 'r')
    timeLi2 = time_from_opar_raw(path_532_raw)
    zlio3t = np.array(lio3t.variables['range'][:]*1000)
    zSelectLi = np.array(zlio3t[np.where(zlio3t < 20000)])
    signal = pd.DataFrame(lio3t.variables['signal'][:,:zSelectLi.shape[0],6] + lio3t.variables['signal'][:,:zSelectLi.shape[0],7])
    # i = signal.where(signal > 0.0).stack().index.values[0][1]
    i = np.where(zSelectLi > 4000)[0][0]
    z_cc2 = zlio3t[i:zSelectLi.shape[0]]
    liSelect = pd.DataFrame(lio3t.variables['signal'][:,i:zSelectLi.shape[0],5], index = timeLi2)
    liSelect_cc = liSelect.iloc[:,40:65].mul(z_cc2[40:65]**2, axis=1).mean(axis=1)
    newdf_cc = new_df.iloc[np.where(new_df.index.get_level_values(1).isin(z_cc2[40:65]))]
    betamol532_cc = np.array(newdf_cc['beta532mol']).reshape((timeLi2.shape[0], z_cc2[40:65].shape[0])).mean(axis=1)
    constK = pd.DataFrame(liSelect_cc / betamol532_cc, index=timeLi2, columns = ["constK"]).astype(np.float64)
    signal_new532 = liSelect.div(constK['constK'], axis = 0).mul(z_cc2**2, axis=1)
    # signal_new532.index = signal_new532.index.strftime("%Y-%m-%d %H:%M").astype("datetime64[ns]")
    betamol532_simu = (new_df.iloc[np.where(new_df.index.get_level_values(1).isin(z_cc2))]['beta532mol'])
    betamol532_simu.index = betamol532_simu.index.set_levels(betamol532_simu.index.levels[0].strftime("%Y-%m-%d %H:%M").astype("datetime64[ns]"), level=0)
    betamol532_simu = betamol532_simu.unstack(level=1)
    betamol532_simu.columns = range(0, z_cc2.shape[0], 1)
    return signal_new532, signal_new355, z_cc, z_cc2, betamol355_simu, betamol532_simu


calibrated355, calibrated532 = pd.DataFrame(), pd.DataFrame()
simulated355, simulated532 = pd.DataFrame(), pd.DataFrame()
li1200_raw_files = [f for f in os.listdir("/home/nmpnguyen/OPAR/LI1200.daily/") if '2019-' in f]
lio3t_raw_files = [f for f in os.listdir("/home/nmpnguyen/OPAR/LIO3T.daily/") if '2019-' in f]
files = np.intersect1d(li1200_raw_files, lio3t_raw_files)
for f in files:
    print(f)
    path_355_raw = "/home/nmpnguyen/OPAR/LI1200.daily/"+f
    path_532_raw = "/home/nmpnguyen/OPAR/LIO3T.daily/"+f
    f = f.split('.')[0]
    path_355_simu = "/homedata/nmpnguyen/OPAR/Processed/LI1200/"+f+"_simul.pkl"
    path_532_simu = "/homedata/nmpnguyen/OPAR/Processed/LIO3T/"+f+"_simul.pkl"
    signal_new532, signal_new355, z_cc, z_cc2, betamol355_simu, betamol532_simu = calibratiion(path_355_raw, path_355_simu, path_532_raw, path_532_simu)
    calibrated355 = pd.concat((calibrated355, signal_new355))
    calibrated532 = pd.concat((calibrated532, signal_new532))
    simulated355 = pd.concat((simulated355, betamol355_simu))
    simulated532 = pd.concat((simulated532, betamol532_simu))


#------------------------------------------------------------------------------------
# time, calendar, units_time = li1200.variables['time'][:], li1200.variables['time'].calendar, li1200.variables['time'].units
# time1 = pd.to_datetime(nc4.num2date(time, units_time, calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True))
# time1 = np.array(time1.strftime("%Y-%m-%d %H:%M")).astype("datetime64[s]")
# time, calendar, units_time = lio3t.variables['time'][:], lio3t.variables['time'].calendar, lio3t.variables['time'].units
# time2 = pd.to_datetime(nc4.num2date(time, units_time, calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True))
# time2 = np.array(time2.strftime("%Y-%m-%d %H:%M")).astype("datetime64[s]")

# t = np.intersect1d(time1, time2)[5].astype("datetime64[s]")
# plt.close()
# Fig, ax = plt.subplots(1,1)
# ax.plot((betamol532_simu.loc[t].astype(np.float64)), z_cc2/1000, '--', color = "green", label = "simulated molecular ATB: 532nm")
# ax.plot((signal_new532.loc[t].astype(np.float64)), z_cc2/1000, color = "green", label = "measured calibrated ATB: 532nm") 
# ax.plot((betamol355_simu.loc[t].astype(np.float64)), z_cc/1000, '--', color = "red", label = "simulated molecular ATB: 355nm")
# ax.plot((signal_new355.loc[t].astype(np.float64)), z_cc/1000, color = "red", label = "measured calibrated ATB: 355nm") 
# ax.set(xlabel = "log10(ATB)", ylabel = "altitude (km)")
# ax.legend()
# plt.tight_layout(rect=[0,0.03,1,0.95])
# Fig.suptitle("LI1200 + LIO3T"+str(t))
# plt.savefig("./test_fig/calibrated2_"+str(t)+".png")


# SR355 = np.empty((0,calibrated355.shape[1]), float); 
# for t in time1:
#     tmp355 = np.array(calibrated355.loc[t]) / np.array(simulated355.loc[t])
#     SR355 = np.append(SR355, tmp355.reshape(1, tmp355.shape[0]), axis=0)


# SR355 = pd.DataFrame(SR355, index = time1)
# SR532 = np.empty((0,calibrated532.shape[1]), float); 
# for t in time2:
#     tmp532 = np.array(calibrated532.loc[t]) / np.array(betamol532_simu.loc[t])
#     SR532 = np.append(SR532, tmp532.reshape(1, tmp532.shape[0]), axis=0)  


# SR532 = pd.DataFrame(SR532, index = time2)
SR355 = calibrated355.div(simulated355, axis=0)
SR532 = calibrated532.div(simulated532, axis=0)

r = np.sort(np.intersect1d(z_cc, z_cc2))
sr355_time = SR355.astype(np.float64).groupby(pd.Grouper(freq = '2min')).mean()
sr532_time = SR532.astype(np.float64).groupby(pd.Grouper(freq = '2min')).mean()
SR355_reshape = sr355_time[np.intersect1d(z_cc, r, return_indices = True)[1]]
SR532_reshape = sr532_time[np.intersect1d(z_cc2, r, return_indices = True)[1]]

time1 = SR355_reshape.index.get_level_values(0)
time2 = SR532_reshape.index.get_level_values(0)
times = np.intersect1d(time1, time2)

SR355plot = np.asarray(SR355_reshape.loc[times].sort_index(ascending=True).stack(dropna=False))
SR532plot = np.asarray(SR532_reshape.loc[times].sort_index(ascending=True).stack(dropna=False))


# SR355plot = np.empty((times.shape[0]*r.shape[0],0), float)
# SR532plot = np.empty((times.shape[0]*r.shape[0],0), float)
# for t in times:
#     tmp355 = np.array(sr355_time.loc[t])
#     tmp532 = np.array(sr532_time.loc[t])
#     SR355_reshape = tmp355[np.intersect1d(z_cc, r, return_indices = True)[1]]
#     SR532_reshape = tmp532[np.intersect1d(z_cc2, r, return_indices = True)[1]]
#     SR532_reshape = SR532_reshape.reshape((r.shape[0], int(SR532_reshape.shape[0]/r.shape[0]))).max(axis=1)
#     # Fig, ax1 = plt.subplots()SR355_reshape
#     # ax1.plot(SR532_reshape, r/1000, color = "green", label = "SR (532nm)")
#     # ax1.plot(SR355_reshape, r/1000, color = "red", label = "SR (355nm)")
#     # ax1.set(xlabel = "SR", ylabel = "altitude (km)")
#     # ax1.legend()
#     # Fig.suptitle("SR new z-resolution and time resolution \n"+str(t))
#     # plt.savefig("./test_fig/SR2lidars_"+str(t)+".png")
#     # plt.close()
#     # plt.clf()
#     SR355plot = np.append(SR355plot, SR355_reshape)
#     SR532plot = np.append(SR532plot, SR532_reshape)


# remove NAN values on these two SR data 
union_nan = np.union1d(np.argwhere(np.isnan(SR355plot)), np.argwhere(np.isnan(SR532plot)))
SR355plot = np.delete(SR355plot, union_nan)
SR532plot = np.delete(SR532plot, union_nan)


from matplotlib import colors
bins = 1000
plt.close()
Fig, axs = plt.subplots()
H = axs.hist2d(SR355plot, SR532plot, bins = bins, vmin= 0, vmax = 30) #norm = colors.NoNorm(vmin=SR355plot.min(), vmax=SR532plot.max())
axs.set(xlabel = "SR (355nm)", ylabel = "SR (532nm)", title = "OPAR 2019, bins = "+bins) #title = pd.to_datetime(times[0]).strftime("%Y-%m-%d")+" bin="+str(bins)
Fig.colorbar(H[3], ax=axs, norm=colors.NoNorm)
axs.set_xlim(0,20)
axs.set_ylim(0,40)
plt.savefig("hist2d1_month.png")

# plt.close()
# Fig, axs = plt.subplots()
# heatmap, xedges, yedges = np.histogram2d(SR355plot, SR532plot, bins = 500)
# plt.imshow(heatmap, origin = 'lower')
