# import xarray as xr 
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
# li1200_raw_files = glob.glob("/home/nmpnguyen/OPAR/LI1200.daily/2019*")
# li1200_simu_files = glob.glob("/homedata/nmpnguyen/OPAR/Processed/LI1200/2019*.pkl")
# lio3t_raw_files = glob.glob("/home/nmpnguyen/OPAR/LIO3T.daily/2019*")
# lio3t_simu_files = glob.glob("/homedata/nmpnguyen/OPAR/Processed/LIO3T/2019*.pkl")

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


# heatmap, xedges, yedges = np.histogram2d(SR355plot, SR532plot, bins = 500)

# PLOT PCOLOR 1 EXAMPLE OF MEASURED CALIBRATED DATA
def plot_calibrated_betamol2(d355, d532, z355, z532):
    import matplotlib.colors as colors
    plt.close()
    Fig, axs = plt.subplots(1, 2, sharey = True)#, figsize=(12,4.8)
    time1 = d355.index.get_level_values(0)
    time2 = d532.index.get_level_values(0)
    plot = axs[0].pcolormesh(time1, z355/1000, (d355.T), norm = colors.LogNorm(), cmap='viridis')#, linthresh=0.1, base=10))
    axs[0].set(ylabel = 'altitude (km)')
    axs[0].xaxis.set_major_formatter(plt_dates.DateFormatter('%H:%M'))
    axs[0].get_xaxis().set_major_locator(plt_dates.HourLocator(interval=1))
    plt.setp(axs[0].get_xticklabels(), rotation = 30)
    axs[0].set_ylim(0,25)
    axs[0].set_title("355 nm")
    Fig.colorbar(plot, ax=axs[0])#.ax.set_ylabel('log10(ATBmol)')
    # colorbar1.ax.set_ylabel('log10(ATBmol)')
    plot = axs[1].pcolormesh(time2, z532/1000, (d532.T), norm = colors.LogNorm(), cmap='viridis')
    axs[1].xaxis.set_major_formatter(plt_dates.DateFormatter('%H:%M'))
    axs[1].get_xaxis().set_major_locator(plt_dates.HourLocator(interval=1))
    plt.setp(axs[1].get_xticklabels(), rotation = 30)
    axs[1].set_ylim(0,20)
    axs[1].set_title("532 nm")
    Fig.colorbar(plot, ax=axs[1]).ax.set_ylabel('(ATB)')
    plt.savefig("/home/nmpnguyen/"+"calibrated"+".png")
    # plt.savefig("/homedata/nmpnguyen/OPAR/Fig/Calibrated_LI1200_LIO3T/"+"calibrated"+f+".png")


def plot_calibrated_betamol1(dataRaw, dataSimul, dataCalib, zLidar, constK, time_plot, lidar_name, channel):
    f, ax = plt.subplots(1,3, sharey = True)
    ax[0].semilogx(dataRaw.loc[time_plot]*(zLidar**2), zLidar/1000)
    ax[1].semilogx(dataSimul.loc[time_plot], zLidar/1000, '--', label = "molecular simulated ATB")
    ax[2].semilogx(dataCalib.loc[time_plot], zLidar/1000, label = "measured calibrated ATB")
    ax[0].set(xlabel = "uncalibrated signal x Z²", ylabel = "Altitude")
    ax[1].set(xlabel = "molecular simulated ATB")
    ax[2].set(xlabel = "measured calibrated ATB")
    ax[2].set_title("Calib.Coef. = %1.3e" %constK.loc[time_plot], loc='right')
    plt.tight_layout(rect=[0,0.03,1,0.95])
    f.suptitle("Calibration step for "+lidar_name+"-"+channel+"\n"+str(time_plot))
    # plt.savefig("/home/nmpnguyen/"+"calibrated_step"+str(time_plot)+".png")
    plt.savefig(get_path_out(main_path = "/homedata/nmpnguyen/OPAR/Fig", time_index = time_plot, lidar_name=lidar_name.upper(), channel=channel, arranging = "day"))


def netcdf_calibrated(signal_to_netcdf, altitude, file_name, lidar_name):
    ncfile = nc4.Dataset("/homedata/nmpnguyen/OPAR/Processed/"+lidar_name.upper()+"/"+file_name+".nc", mode='w', format = 'NETCDF4')
    alt_dim = ncfile.createDimension('alt', altitude.shape[0])
    time_dim = ncfile.createDimension('time', signal_to_netcdf.index.get_level_values(0).shape[0])
    print(ncfile)
    ncfile.title = file_name+'data'
    ncfile.subtitle = 'opar data'
    alt = ncfile.createVariable('alt', np.float64, ('alt',))
    alt.units = 'm'
    alt.long_name = 'altitude'
    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = 'seconds since 2000-01-01'
    time.long_name = 'time'
    data = ncfile.createVariable('data', np.float64, ('time', 'alt'))
    data.units = 'm−1.sr−1'
    data.standard_name = 'Attenuated Backscatter Signal'
    print(data)
    alt[:] = altitude
    # time[:] = nc4.date2num((time1).tolist(), time.units).astype(np.int64)
    time[:] = nc4.date2num(signal_to_netcdf.index.get_level_values(0).tolist(), time.units).astype(np.int64)
    data[:,:] = signal_to_netcdf
    return ncfile


def calibration(path_raw, path_simu, file_name, lidar_name):    
    def time_from_opar_raw(path):
        data = nc4.Dataset(path, 'r')
        time, calendar, units_time = data.variables['time'][:], data.variables['time'].calendar, data.variables['time'].units
        timeLidar = pd.to_datetime(nc4.num2date(time, units_time, calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True))
        # timeLidar = np.array(timeLidar.strftime("%Y-%m-%d %H:%M")).astype("datetime64[s]")
        return timeLidar        
    def get_signal_significatif(signal):
        array_nozero = np.where(signal)
        if len(array_nozero[0])/(signal.shape[0]*signal.shape[1]) > 0.1:
            return signal
        else:
            return None 
    new_df = pd.read_pickle(path_simu)
    li1200 = nc4.Dataset(path_raw, 'r')
    timeLi1 = time_from_opar_raw(path_raw)
    rangeLi = li1200.variables['range'][:][np.where(li1200.variables['range'][:]<=100)]
    altLi = rangeLi + 2.1
    zli1200 = altLi*1000 #np.array(li1200.variables['range'][:]*1000)+2100
    rSelectLi = np.array(rangeLi[np.where(rangeLi < 25)])
    zSelectLi = np.array(zli1200[np.where(rangeLi < 25)])
    if lidar_name == "li1200":
        signal_before = (li1200.variables['signal'][:,:rangeLi.shape[0],5])
        fc = li1200.variables['signal'][:,np.where(rangeLi>80)[0],5]
        mean_fc = fc.mean(axis =1).reshape(-1, 1)
        beta = "beta355mol"
        channel = li1200.variables['channel'][5]
    else:
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
        signal = signal_before-mean_fc
        i = np.where(rangeLi > 4)[0][0]
        r_cc = rSelectLi[i:rSelectLi.shape[0]]
        z_cc = zSelectLi[i:rSelectLi.shape[0]]
        # liSelect = pd.DataFrame(li1200.variables['signal'][:,i:zSelectLi.shape[0],5], index = timeLi1)
        liSelect = pd.DataFrame(signal[:,i:rSelectLi.shape[0]], index = timeLi1)
        liSelect_cc = liSelect.iloc[:,:8].mul(z_cc[:8]**2, axis=1).mean(axis=1)
        newdf_cc = new_df.iloc[np.where((new_df.index.get_level_values(1)).isin(r_cc[:8]))]
        betamol355_cc = newdf_cc[beta].unstack(level=1).mean(axis=1) # mean each day 
        # betamol532_cc = np.array(newdf_cc['beta532mol']).reshape((time.shape[0], 10)).mean(axis=1)
        constK1 = pd.DataFrame(liSelect_cc / betamol355_cc, columns = ["constK"]).astype(np.float64)
        signal_new355 = pd.DataFrame(signal[:,:rSelectLi.shape[0]], index=timeLi1).mul(zSelectLi**2, axis=1).div(constK1['constK'], axis = 0)
        print(signal_new355.shape)
        # signal_new355.index = signal_new355.index.strftime("%Y-%m-%d %H:%M").astype("datetime64[ns]")
        betamol355_simu = new_df.iloc[np.where((new_df.index.get_level_values(1)).isin(rSelectLi))][beta]
        # betamol355_simu.index = betamol355_simu.index.set_levels(betamol355_simu.index.levels[0].strftime("%Y-%m-%d %H:%M").astype("datetime64[ns]"), level=0)
        betamol355_simu = betamol355_simu.unstack(level=1)
        print(betamol355_simu.shape)
        betamol355_simu.columns = range(0, zSelectLi.shape[0], 1)
        # time_plot = timeLi1[5]
        # print('plot ATB')
        # plot_calibrated_betamol1(pd.DataFrame(signal, index=timeLi1), betamol355_simu, signal_new355, zSelectLi, constK1, time_plot, lidar_name, channel)        
    li1200.close()
    return signal_new355, zSelectLi, betamol355_simu, constK1#, signal_new532, z_cc2, betamol532_simu, constK2


calibrated355, calibrated532 = pd.DataFrame(), pd.DataFrame()
simulated355, simulated532 = pd.DataFrame(), pd.DataFrame()
K1, K2 = pd.DataFrame(), pd.DataFrame()
li1200_raw_files = [f for f in os.listdir("/home/nmpnguyen/OPAR/LI1200.daily/") if '2019-' in f]
lio3t_raw_files = [f for f in os.listdir("/home/nmpnguyen/OPAR/LIO3T.daily/") if '2019-' in f]
files = np.intersect1d(li1200_raw_files, lio3t_raw_files)
files = np.delete(files, np.where(np.isin(files, ['2019-02-19.nc4', '2019-11-19.nc4', '2019-06-18.nc4', '2019-05-20.nc4'])))
numb_profil_532 = []; numb_profil_355 = []
sky_clear_532 = []; sky_clear_355 = []
li1200_raw_files.sort()
lio3t_raw_files.sort()
# for f in files[:5]:
# for f1 in li1200_raw_files[:2]:
#     path_raw = "/home/nmpnguyen/OPAR/LI1200.daily/"+f1
#     f1 = f1.split('.')[0]
#     path_simu = "/homedata/nmpnguyen/OPAR/Processed/LI1200/"+f1+"_simul.pkl"
#     signal_new355, z_cc, betamol355_simu, constK1 = calibration(path_raw, path_simu, f1, "li1200")
#     print(path_raw)
#     if constK1 is None:
#         sky_clear_355.append(pd.to_datetime(f1))
#     else:
#         # netcdf_calibrated(signal_new355, z_cc, file_name = f1+"_Low_calibrated", lidar_name = "li1200")
#         # netcdf_calibrated(betamol355_simu, z_cc, file_name = f1+"_Low_simulated", lidar_name = "li1200")
#         numb_profil_355.append(pd.to_datetime(f1))
#         Sr355 = signal_new355.div(betamol355_simu)
#         sr355_time = Sr355.astype(np.float64).groupby(pd.Grouper(freq = '2min')).mean()
#     K1 = pd.concat((K1, constK1))
#     calibrated355 = pd.concat((calibrated355, sr355_time))


# for f2 in lio3t_raw_files[5:6]:
#     path_raw = "/home/nmpnguyen/OPAR/LIO3T.daily/"+f2    
#     f2 = f2.split('.')[0]    
#     path_simu = "/homedata/nmpnguyen/OPAR/Processed/LIO3T/"+f2+"_simul.pkl"    
#     signal_new532, z_cc2, betamol532_simu, constK2 = calibration(path_raw, path_simu, f2, "lio3t")  
#     print(path_raw)  
#     if constK2 is None:
#         sky_clear_532 = sky_clear_532.append(pd.to_datetime(f2))
#     else:
#         # netcdf_calibrated(signal_new532, z_cc2, file_name = f2+"_calibrated", lidar_name = "lio3t")
#         # netcdf_calibrated(betamol532_simu, z_cc2, file_name = f2+"_simulated", lidar_name = "lio3t") 
#         numb_profil_532.append(pd.to_datetime(f2))
#         Sr532 = signal_new532.div(betamol532_simu)
#         sr532_time = Sr532.astype(np.float64).groupby(pd.Grouper(freq = '2min')).mean()   
#     K2 = pd.concat((K2, constK2))
#     calibrated532 = pd.concat((calibrated532, sr532_time))


# K1.to_pickle("/homedata/nmpnguyen/OPAR/Processed/K1.pkl")
# K2.to_pickle("/homedata/nmpnguyen/OPAR/Processed/K2.pkl")


def test_sr(sr355, sr532, z1, z2):
    r = np.sort(np.intersect1d(z1, z2))
    alt_real = r[np.where((r>=6000)&(r<=20000))]
    sr355_reshape = sr355[np.intersect1d(z1, alt_real, return_indices = True)[1]]
    sr532_reshape = sr532[np.intersect1d(z2, alt_real, return_indices = True)[1]]
    sr355_reshape.drop_duplicates(keep='first', inplace=True)
    sr532_reshape.drop_duplicates(keep='first', inplace=True)
    time1 = sr355_reshape.index.get_level_values(0)
    time2 = sr532_reshape.index.get_level_values(0)
    times = np.sort(np.intersect1d(time1, time2))
    SR355plot = np.asarray(sr355_reshape.loc[times].sort_index(ascending=True).stack(dropna=False))
    SR532plot = np.asarray(sr532_reshape.loc[times].sort_index(ascending=True).stack(dropna=False))
    for t in times:
        plt.close()
        ff, ax = plt.subplots()
        ax.plot(sr532_reshape.loc[t], alt_real, color = "green", label="SR532")
        ax.plot(sr355_reshape.loc[t], alt_real, color = "red", label="SR355")
        ax.legend()
        ax.set(xlabel = "SR", ylabel="Altitude")
        # plt.savefig("/homedata/nmpnguyen/OPAR/Fig/Calibrated_LI1200_LIO3T/"+"SR_profil"+str(t)+".png")
        plt.savefig(get_path_out(main_path="/homedata/nmpnguyen/OPAR/Fig/", time_index=t, lidar_name="LI1200", channel="SRProfil", arranging = "day"))
        plt.savefig(get_path_out(main_path="/homedata/nmpnguyen/OPAR/Fig/", time_index=t, lidar_name="LIO3T", channel="SRProfil", arranging = "day"))
    # remove NAN values on these two SR data 
    SR355plot[(SR355plot == np.inf)|(SR355plot == -np.inf)] = np.nan
    SR532plot[(SR532plot == np.inf)|(SR532plot == -np.inf)] = np.nan
    union_nan = np.union1d(np.argwhere(np.isnan(SR355plot)), np.argwhere(np.isnan(SR532plot)))
    SR355plot = np.delete(SR355plot, union_nan)
    SR532plot = np.delete(SR532plot, union_nan)
    return SR355plot, SR532plot
    
# files = ["2019-01-21.nc4", "2019-01-24.nc4"]
from matplotlib import colors  
calibrated355, calibrated532 = pd.DataFrame(), pd.DataFrame() 
for f in files:
    path_raw = "/home/nmpnguyen/OPAR/LI1200.daily/"+f
    f1 = f.split('.')[0]
    path_simu = "/homedata/nmpnguyen/OPAR/Processed/LI1200/"+f1+"_simul.pkl"
    signal_new355, z_cc, betamol355_simu, constK1 = calibration(path_raw, path_simu, f1, "li1200")
    print(path_raw)
    path_raw = "/home/nmpnguyen/OPAR/LIO3T.daily/"+f    
    f2 = f.split('.')[0]    
    path_simu = "/homedata/nmpnguyen/OPAR/Processed/LIO3T/"+f2+"_simul.pkl"    
    signal_new532, z_cc2, betamol532_simu, constK2 = calibration(path_raw, path_simu, f2, "lio3t")  
    print(path_raw)  
    Sr355 = signal_new355.div(betamol355_simu)
    sr355_time = Sr355.astype(np.float64).groupby(pd.Grouper(freq = '2min')).mean()
    Sr532 = signal_new532.div(betamol532_simu)
    sr532_time = Sr532.astype(np.float64).groupby(pd.Grouper(freq = '2min')).mean()   
    SR355plot, SR532plot = test_sr(sr355_time, sr532_time, z_cc, z_cc2)
    bins = 100
    plt.close()
    plt.clf()
    plt.cla()
    Fig, axs = plt.subplots()
    H = axs.hist2d(SR355plot, SR532plot, bins = bins, vmax=20, vmin=0)#, norm=colors.LogNorm())# vmin= 0, vmax = 30) 
    axs.set(ylabel = "SR (532nm)", xlabel = "SR (355nm)", title = f+" bins = "+str(bins)) #title = pd.to_datetime(times[0]).strftime("%Y-%m-%d")+" bin="+str(bins)
    plt.colorbar(H[3], ax=axs)#, norm=colors.NoNorm    
    # axs.set_xlim(0,20)    
    # axs.set_ylim(0,40)
    plt.savefig("/homedata/nmpnguyen/OPAR/Fig/Calibrated_LI1200_LIO3T/SR2noCouchesBasses_"+f+".png")
    calibrated355 = pd.concat((calibrated355, sr355_time))
    calibrated532 = pd.concat((calibrated532, sr532_time))


calibrated355.to_pickle("/homedata/nmpnguyen/OPAR/Processed/SR355.pkl")
calibrated532.to_pickle("/homedata/nmpnguyen/OPAR/Processed/SR532.pkl")
SR355Plot, SR532Plot = test_sr(calibrated355, calibrated532, z_cc, z_cc2)
pd.DataFrame(SR355Plot).to_pickle("/homedata/nmpnguyen/OPAR/Processed/SR355_split.pkl")
pd.DataFrame(SR532Plot).to_pickle("/homedata/nmpnguyen/OPAR/Processed/SR532_split.pkl")

# # distributoin plot of Calibration coef
# plt.close()
# # bins_values = np.arange(start = pd.DataFrame(K2/(10**16)).min().constK, stop = pd.DataFrame(K2/(10**16)).max().constK, step = 0.05)
# fig, ax = plt.subplots(ncols = 2, sharey=False)
# n, bins, patches = ax[1].hist(K2.iloc[np.where(K2['constK']<=0.5e17)[0]]['constK'], 50, density=False, label="532nm")
# # n, bins, patches = ax[1].hist(K2['constK'], 50, density=False, label= "532nm")
# ax[1].legend()
# ax[1].set_xlim(0e17, 0.5e17)
# n1, bins1, patches1 = ax[0].hist(K1['constK'], 50, density=False, label = "355nm")
# ax[0].legend()
# ax[0].set(ylabel="Frequency")
# plt.suptitle("Distribution of Calibration Coef. (2019)")
# plt.tight_layout()
# fig.savefig("/home/nmpnguyen/distribution_K_global.png")
# plt.show()


# # Scatter plot of Calibration coef.
# fig, ax = plt.subplots(2,1, sharex = True)
# ax[0].scatter(K1.index.tolist(), K1['constK'], s=1.0, c="red", alpha=0.5)
# ax[1].scatter(K2.index.tolist(), K2['constK'], s=1.0, c="green", alpha=0.5)
# ax[1].xaxis.set_major_formatter(plt_dates.DateFormatter('%y-%m-%d %H:%M'))
# plt.setp(ax[1].get_xticklabels(), rotation = 30)
# ax[1].set_xlim(0e17, 0.5e17)
# fig.suptitle("Calibration coef.")
# plt.tight_layout()
# plt.savefig("./test_fig/constK.png")

# # plot number of days when have measurements per month 
# table_daily = pd.DataFrame(data=0, index=range(1,13), columns=["numb_profil_355", "numb_profil_532", "sky_clear_532", "sky_clear_355"])
# for i,j,k,l in zip(numb_profil_355,numb_profil_532,sky_clear_532,sky_clear_355):
#     print(i, j, k, l)
# numb_profil_355 = pd.DataFrame(numb_profil_355)
# for i in range(0, numb_profil_355.shape[0]):
#     print(numb_profil_355.loc[i,0].month)
#     table_daily.loc[numb_profil_355.loc[i,0].month, "numb_profil_355"] += 1

# numb_profil_532 = pd.DataFrame(numb_profil_532)
# for i in range(0, numb_profil_532.shape[0]):
#     print(numb_profil_532.loc[i,0].month)
#     table_daily.loc[numb_profil_532.loc[i,0].month, "numb_profil_532"] += 1 

# sky_clear_532 = pd.DataFrame(sky_clear_532)
# for i in range(0, sky_clear_532.shape[0]):
#     print(sky_clear_532.loc[i,0].month)
#     table_daily.loc[sky_clear_532.loc[i,0].month, "sky_clear_532"] += 1 

# sky_clear_355 = pd.DataFrame(sky_clear_355)
# for i in range(0, sky_clear_355.shape[0]):
#     print(sky_clear_355.loc[i,0].month)
#     table_daily.loc[sky_clear_355.loc[i,0].month, "sky_clear_355"] += 1

# # numb_profil_355 = pd.DataFrame(numb_profil_355)
# # numb_profil_532 = pd.DataFrame(numb_profil_532)
# # profil355freq = numb_profil_355.groupby(numb_profil_355[0].dt.month).agg('count')
# # profil532freq = numb_profil_532.groupby(numb_profil_532[0].dt.month).agg('count')
# # sky_clear_532 = pd.DataFrame(sky_clear_532)
# # sky_clear_355 = pd.DataFrame(sky_clear_355)
# # clear355freq = pd.merge(left=pd.DataFrame(index=range(1,13)), right=sky_clear_355.groupby(sky_clear_355[0].dt.month).agg('count'),
# #     left_index=True, copy=True, how="left")
# # clear532freq = sky_clear_532.groupby(sky_clear_532[0].dt.month).agg('count') # any file LIO3T in 2019 is None ???
# plt.clf()
# plt.cla()
# f, ax= plt.subplots()
# barWidth = 0.25
# plt.bar(table_daily.index, table_daily["numb_profil_355"], label = "355nm", width=barWidth, color = "red", edgecolor='white', alpha = 0.75)
# plt.bar(table_daily.index, table_daily["sky_clear_355"], label = "355nm(clear sky)", bottom = table_daily["numb_profil_355"], width=barWidth, color = "red", edgecolor='white', alpha = 0.5, hatch = "x")
# plt.bar(table_daily.index+barWidth, table_daily["numb_profil_532"], label = "532nm", width=barWidth, color = "green", edgecolor='white', alpha = 0.75)
# plt.bar(table_daily.index+barWidth, table_daily["sky_clear_532"], label = "532nm(clear sky)", bottom = table_daily["numb_profil_532"], width=barWidth, color = "green", edgecolor='white', alpha = 0.5, hatch = "x")
# plt.xlabel('Months')
# # plt.xticks(['Fev', 'April', 'June', 'Aug', 'Oct', 'Dec'])
# plt.ylabel('Frequency')
# plt.title("Number of measurements days per month in 2019")
# plt.legend()
# plt.savefig("/home/nmpnguyen/daily_profils_per_month.png")
# plt.show()

# # plot number of profils in each months 
# profil355freq = K1.resample("M").count() # car K1 est DataFrame de DateIndex
# profil532freq = K2.resample("M").count() # car K1 est DataFrame de DateIndex
# f, ax= plt.subplots()
# barWidth = 0.25
# plt.bar(profil355freq.index.month, profil355freq['constK'], label = "355nm", width=barWidth, color = "red", edgecolor='white', alpha = 0.5)
# plt.bar(profil532freq.index.month+barWidth, profil532freq['constK'], label = "532nm", width=barWidth, color = "green", edgecolor='white', alpha = 0.5) 
# plt.xlabel('Months')
# # plt.xticks(['Fev', 'April', 'June', 'Aug', 'Oct', 'Dec'])
# plt.ylabel('Frequency')
# plt.title("Profils per month in 2019")
# plt.legend()
# plt.savefig("/home/nmpnguyen/profils_per_month.png")
# plt.show()