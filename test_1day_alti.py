import sys
sys.path.append('/home/nmpnguyen/Codes/')
import remove_error_file

import xarray as xr
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
from matplotlib import colors



def get_path_out(main_path, filename, time_index, lidar_name, channel, arranging = False):
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
        file_name_out = os.path.basename(filename).split(".")[0] + "_" + channel + ".png"
        path_file_out = main_path + '/' + lidar_name.upper() + '/' + year + "/" + file_name_out
    print(path_file_out)
    return path_file_out



def plot_calibrated_betamol1(dataRaw, dataSimul, dataCalib, zLidar, rangeLi, constK, time_plot, lidar_name, channel, filename):
    f, ax = plt.subplots(1,3, sharey = True)
    ax[0].semilogx(dataRaw.loc[time_plot], zLidar/1000)
    ax[0].set(xlabel = "signal raw", ylabel = "Altitude, km")
    # ax[1].semilogx(dataRaw.loc[time_plot]*(rangeLi**2), zLidar/1000)
    # ax[1].set(xlabel="signal raw x range^2")
    ax[2].semilogx(dataCalib.loc[time_plot], zLidar/1000, label = "measured calibrated")
    ax[2].semilogx(dataSimul.loc[time_plot], zLidar/1000, '--', label = "molecular simulated")
    ax[2].set(xlabel = "ATB")
    ax[2].set_title("Calib.Coef. = %1.3e" %constK.loc[time_plot], loc='right')
    ax[2].legend(loc="lower center")
    # ax[2].set_xlim(1e-8, 1.5e-6)
    # ax[1].plot(dataCalib.loc[time_plot]/dataSimul.loc[time_plot], zLidar/1000)
    # ax[1].vlines(1, ymin=0, ymax=25, linestyles="--", color="red")
    ax[1].semilogx(dataRaw.loc[time_plot]*(rangeLi**2), zLidar/1000)
    ax[1].set(xlabel="signal raw x r2")
    f.suptitle("Calibration step for "+lidar_name+"-"+channel+"\n"+str(time_plot))
    plt.xticks()
    plt.tight_layout()
    # plt.savefig("/home/nmpnguyen/"+"calibrated_step"+str(time_plot)+".png")
    plt.savefig(get_path_out(main_path = "/homedata/nmpnguyen/OPAR/Fig", filename = filename,time_index = time_plot, lidar_name=lidar_name.upper(), channel=channel+"_l2", arranging = "day"))
    plt.close(f)


def calibration(path_raw, path_simu, file_name, lidar_name, opts_plot, channel_numb, alti_calib):         
    def get_signal_significatif(signal):
        array_nozero = np.where(signal)
        if len(array_nozero[0])/(signal.shape[0]*signal.shape[1]) > 0.1:
            return signal
        else:
            return None 
    new_df = pd.read_pickle(path_simu)
    li1200 = xr.open_dataset(path_raw)
    timeLi1 = li1200.time
    rangeLi = li1200.variables['range'][:][li1200.variables['range'][:]<=100]
    altLi  = rangeLi + 2.1
    zli1200 = altLi*1000 
    rSelectLi = np.array(rangeLi[np.where(altLi < 25)])*1000 #en m
    zSelectLi = np.array(zli1200[np.where(altLi < 25)]) #en m
    if lidar_name == "li1200":
        print("step1")
        signal_before = (li1200.variables['signal'][:,:rangeLi.shape[0],channel_numb])
        fc = li1200.variables['signal'][:,np.where(rangeLi>80)[0],channel_numb]
        mean_fc = fc.mean(axis =1).reshape(-1, 1)
        beta = "beta355mol"
        channel = li1200.variables['channel'][channel_numb]
    else:
        print("step1")
        signal_before = (li1200.variables['signal'][:,:rangeLi.shape[0],6] + li1200.variables['signal'][:,:rangeLi.shape[0],7])
        fc = li1200.variables['signal'][:,np.where(rangeLi>80)[0],6] + li1200.variables['signal'][:,np.where(rangeLi>80)[0],7]
        mean_fc = fc.mean(axis =1).reshape(-1, 1)
        beta = "beta532mol"
        channel = li1200.variables['channel'][6]+li1200.variables['channel'][7]
    #--------------------------
    if get_signal_significatif(signal_before) is None:
        print("Signal is None")
        signal_new355 = None
        z_cc = None
        betamol355_simu = None
        constK1 = None
    else:
        print("step2")
        signal = signal_before-mean_fc
        i = np.where(altLi >= alti_calib/1000)[0][0]
        print("Altitude ------------")
        print(i, altLi[i], alti_calib)
        r_cc = rSelectLi[i:rSelectLi.shape[0]]
        z_cc = zSelectLi[i:zSelectLi.shape[0]]
        print(r_cc) #en km
        print(z_cc) #en m
        # liSelect = pd.DataFrame(li1200.variables['signal'][:,i:zSelectLi.shape[0],5], index = timeLi1)
        liSelect = pd.DataFrame(signal[:,i:rSelectLi.shape[0]], index = timeLi1)
        liSelect_cc = liSelect.iloc[:,:8].mul(r_cc[:8]**2, axis=1).mean(axis=1)
        newdf_cc = new_df.iloc[np.where((new_df.index.get_level_values(1)).isin(z_cc[:8]))]
        betamol355_cc = newdf_cc[beta].unstack(level=1).mean(axis=1) # mean each day 
        # betamol532_cc = np.array(newdf_cc['beta532mol']).reshape((time.shape[0], 10)).mean(axis=1)
        constK1 = pd.DataFrame(liSelect_cc / betamol355_cc, columns = ["constK"]).astype(np.float64)
        print("step3")
        signal_new355 = pd.DataFrame(signal[:,:rSelectLi.shape[0]], index=timeLi1).mul(rSelectLi**2, axis=1).div(constK1['constK'], axis = 0)
        # signal_new355.index = signal_new355.index.strftime("%Y-%m-%d %H:%M").astype("datetime64[ns]")
        betamol355_simu = new_df.iloc[np.where((new_df.index.get_level_values(1)).isin(zSelectLi))][beta]
        # betamol355_simu.index = betamol355_simu.index.set_levels(betamol355_simu.index.levels[0].strftime("%Y-%m-%d %H:%M").astype("datetime64[ns]"), level=0)
        betamol355_simu = betamol355_simu.unstack(level=1)
        betamol355_simu.columns = range(0, zSelectLi.shape[0], 1)
        plt.cla()
        plt.clf()
        if opts_plot:
            print('plot ATB')
            for time_plot in timeLi1:#time_plot = timeLi1[5]
                plot_calibrated_betamol1(pd.DataFrame(signal[:,:zSelectLi.shape[0]], index=timeLi1), betamol355_simu, signal_new355, zSelectLi, rSelectLi, constK1, time_plot, lidar_name, channel=channel, filename=file_name)        
    li1200.close()
    return signal_new355, zSelectLi, betamol355_simu, constK1, channel#, signal_new532, z_cc2, betamol532_simu, constK2


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


def plot_ATB_SR(signal_new, betamol_simu, z, time, fn, lidar_name, channel):
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots()
    p0 = ax.pcolormesh(time, z/1000, signal_new.T, shading = "nearest", norm = colors.LogNorm())
    colorbar = plt.colorbar(p0, ax=ax)
    plt.setp(ax.get_xticklabels(), rotation = 90)
    ax.set(ylabel="Alt, km")    
    colorbar.set_label("ATB, m-1.sr-1")
    plt.suptitle(lidar_name.upper()+fn+"\n Mesured calibrated ATB")
    plt.tight_layout()
    # plt.savefig("/homedata/nmpnguyen/OPAR/Fig/Calibrated_LI1200_LIO3T/"+fn+"_"+lidar_name.upper()+"_calibAlti.png")
    plt.savefig(get_path_out(main_path = "/homedata/nmpnguyen/OPAR/Fig/Calibrated_LI1200_LIO3T/", filename=fn, time_index=time[0], lidar_name=lidar_name, channel=channel+"_pcolor_l2", arranging = ""))
    plt.close(fig)

    
def Processed(fn, opts_plot):    
    # fn = "2019-01-24"
    li1200 = "/home/nmpnguyen/OPAR/LI1200.daily/"+fn+".nc4"
    lio3t = "/home/nmpnguyen/OPAR/LIO3T.daily/"+fn+".nc4"
    #----------------------------------
    # calibrer_v1.
    #----------------------------------
    path_simu = "/homedata/nmpnguyen/OPAR/Processed/LIO3T/"+fn+"_simul.pkl"
    signal_new532, z_cc2, betamol532_simu, constK2, channel2 = calibration(lio3t, path_simu, fn, "lio3t", opts_plot, None) 
    signal_new532.columns = z_cc2
    t532 = pd.to_datetime(signal_new532.index)
    signal_new532.to_pickle("/homedata/nmpnguyen/OPAR/Processed/LIO3T/"+fn+"_"+channel2+"_calibrated.pkl")
    sr532 = (signal_new532.div(betamol532_simu)).astype("float64")
    sr532.to_pickle("/homedata/nmpnguyen/OPAR/Processed/LIO3T/"+fn+"_"+channel2+"_SR.pkl")
    #----------------------------------
    path_simu = "/homedata/nmpnguyen/OPAR/Processed/LI1200/"+fn+"_simul.pkl"
    signal_new355, z_cc, betamol355_simu, constK1, channel1 = calibration(li1200, path_simu, fn, "li1200", opts_plot, 4)
    signal_new355.columns = z_cc
    t355 = pd.to_datetime(signal_new355.index)
    signal_new355.to_pickle("/homedata/nmpnguyen/OPAR/Processed/LI1200/"+fn+"_"+channel1+"_calibrated.pkl")
    sr355 = (signal_new355.div(betamol355_simu)).astype("float64")
    sr355.to_pickle("/homedata/nmpnguyen/OPAR/Processed/LI1200/"+fn+"_"+channel1+"_SR.pkl")
    #----------------------------------
    path_simu = "/homedata/nmpnguyen/OPAR/Processed/LI1200/"+fn+"_simul.pkl"
    signal_new355, z_cc, betamol355_simu, constK1, channel1 = calibration(li1200, path_simu, fn, "li1200", opts_plot, 5)
    signal_new355.columns = z_cc
    t355 = pd.to_datetime(signal_new355.index)
    signal_new355.to_pickle("/homedata/nmpnguyen/OPAR/Processed/LI1200/"+fn+"_"+channel1+"_calibrated.pkl")
    sr355 = (signal_new355.div(betamol355_simu)).astype("float64")
    sr355.to_pickle("/homedata/nmpnguyen/OPAR/Processed/LI1200/"+fn+"_"+channel1+"_SR.pkl")
    #----------------------------------
    # plot_ATB_SR(signal_new355, betamol355_simu, z_cc, t355, fn, "LI1200")
    # plot_ATB_SR(signal_new532, betamol532_simu, z_cc2, t532, fn, "LIO3T")
    #----------------------------------
    
    # #----------------------------------
    # print("plot ATB 2 lidars 1 plot")
    # plt.close()
    # f, ax = plt.subplots(1,3, sharey=True)
    # ax[0].plot(signal_new355.loc[t355[10]], z_cc/1000, color="red", label="355 (L)")
    # ax[0].plot(betamol355_simu.loc[t355[10]], z_cc/1000, "--", color="black", label="355 mol(L)")
    # ax[0].legend()
    # ax[0].set(xlabel="ATB", ylabel="alt,km")
    # ax[0].set_ylim(0.0, 25.0)
    # ax[1].plot(signal_new532.loc[t532[10]], z_cc2/1000, color="green", label="532(p)")
    # ax[1].plot(betamol532_simu.loc[t532[10]], z_cc2/1000, "--", color="black", label="532 mol(p)")
    # ax[1].legend()
    # ax[1].set(xlabel="ATB", ylabel="alt,km")
    # ax[2].plot(sr355.loc[t355[10]], z_cc/1000+2.1, color="red", label="355,L")
    # ax[2].plot(sr532.loc[t532[10]], z_cc2/1000+2.1, color="green", label="532,p")
    # ax[2].set(xlabel="SR")
    # ax[2].legend()
    # plt.suptitle("OPAR Example LI1200 and LIO3T \n"+fn)
    # plt.tight_layout()
    # plt.savefig("/homedata/nmpnguyen/OPAR/Fig/Calibrated_LI1200_LIO3T/"+fn+"_calibAlti.png")
    # #----------------------------------
    # print("SR hist")
    # SR355plot, SR532plot = test_sr(sr355, sr532, z_cc, z_cc2)
    # bins = 100
    # plt.close()
    # Fig, axs = plt.subplots()
    # h = axs.hist2d(SR355plot, SR532plot, bins = bins, vmin= 0, vmax = 30)# , norm=colors.LogNorm())# ) 
    # axs.set(ylabel = "SR (532nm)", xlabel = "SR (355nm)", title = fn+" bins = "+str(bins) + " (alti)") #title = pd.to_datetime(times[0]).strftime("%Y-%m-%d")+" bin="+str(bins)
    # plt.colorbar(h[3], ax=axs)#, norm=colors.NoNorm
    # plt.savefig("/homedata/nmpnguyen/OPAR/Fig/Calibrated_LI1200_LIO3T/"+fn+"_SRalti.png")
    # #----------------------------------
    # ATB355, ATB532 = test_sr(signal_new355, signal_new532, z_cc, z_cc2)
    # bins = 100
    # fig, ax = plt.subplots()
    # h = ax.hist2d(ATB355, ATB532, bins=bins, norm = colors.LogNorm())
    # ax.set(xlabel="ATB (355nm)", ylabel="ATB (532nm)", title= fn+" bins=" + str(bins)+ " (alti)")
    # plt.colorbar(h[3], ax=ax)
    # plt.savefig("/homedata/nmpnguyen/OPAR/Fig/Calibrated_LI1200_LIO3T/"+fn+"_ATBalti.png")

def Processed_532(main_path, lidar_name ,fn, opts_plot, alti_calib):    
    lidar = main_path+lidar_name.upper()+".daily/"+fn+".nc4"
    #----------------------------------
    path_simu = "/homedata/nmpnguyen/OPAR/Processed/"+lidar_name.upper()+"/"+fn+"_simul.pkl"
    signal_new532, z_cc2, betamol532_simu, constK2, channel2 = calibration(lidar, path_simu, fn, lidar_name.lower(), opts_plot, None, alti_calib) 
    if signal_new532 is not None:
        t532 = pd.to_datetime(signal_new532.index)
        signal_new532.to_pickle("/homedata/nmpnguyen/OPAR/Processed/LIO3T/"+fn+"_"+channel2+"_ATB.pkl")
        sr532 = (signal_new532.div(betamol532_simu)).astype("float64")
        sr532.columns = z_cc2
        sr532.to_pickle("/homedata/nmpnguyen/OPAR/Processed/LIO3T/"+fn+"_"+channel2+"_SR.pkl")
        #----------------------------------
        plot_ATB_SR(signal_new532, betamol532_simu, z_cc2, t532, fn, lidar_name, channel2)


def Processed_355(main_path, lidar_name ,fn, opts_plot, channel_numb, alti_calib):    
    lidar = main_path+lidar_name.upper()+".daily/"+fn+".nc4"
    #----------------------------------
    path_simu = "/homedata/nmpnguyen/OPAR/Processed/"+lidar_name.upper()+"/"+fn+"_simul.pkl"
    signal_new355, z_cc, betamol355_simu, constK1, channel1 = calibration(lidar, path_simu, fn, lidar_name.lower(), opts_plot, channel_numb, alti_calib)
    if signal_new355 is not None: 
        t355 = pd.to_datetime(signal_new355.index)
        signal_new355.to_pickle("/homedata/nmpnguyen/OPAR/Processed/LI1200/"+fn+"_"+channel1+"_ATB.pkl")
        sr355 = (signal_new355.div(betamol355_simu)).astype("float64")
        sr355.columns = z_cc
        sr355.to_pickle("/homedata/nmpnguyen/OPAR/Processed/LI1200/"+fn+"_"+channel1+"_SR.pkl")
        #----------------------------------
        plot_ATB_SR(signal_new355, betamol355_simu, z_cc, t355, fn, lidar_name, channel1)



from argparse import Namespace, ArgumentParser
def main():
    parser = ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, help="Main folder of OPAR data, included LI1200.daily and LIO3T.daily folders", required=True)
    parser.add_argument("--day", "-d", type=str, help = "YYYY-MM-DD daily file")
    parser.add_argument("--plot", "-p", type=lambda x: x.lower()=='true', default=True, help="To create plots")
    parser.add_argument("--lidar_name", "-l", type=str, help="Name of lidar on upper character", required=True)
    parser.add_argument("--altitude", "-a", type=int, help="Calibration altitude (m)", required=True)
    opts = parser.parse_args()
    print(opts)
    if opts.day is None:
        m=0
        if opts.lidar_name.upper() == "LIO3T":
            print(opts.folder +"/LIO3T.daily/"+ "2019*.nc4")
            os.chdir(opts.folder +"/LIO3T.daily/")
            lf = remove_error_file.remove_error_file(opts.folder, opts.lidar_name, 6)
            for fn in lf:
                print(fn)
                m+=1
                fn = fn.split(".")[0]
                Processed_532(opts.folder, opts.lidar_name ,fn, opts.plot, opts.altitude)
        else:
            print(opts.folder +"/LI1200.daily/"+ "2019*.nc4")
            os.chdir(opts.folder +"/LI1200.daily/")
            lf = remove_error_file.remove_error_file(opts.folder, opts.lidar_name, 5)
            for fn in lf:               
                print(fn)
                fn = fn.split(".")[0]
                # Processed_355(opts.folder, opts.lidar_name ,fn, opts.plot,4, opts.altitude)
                Processed_355(opts.folder, opts.lidar_name ,fn, opts.plot,5, opts.altitude)
    else:
        fn = opts.day
        if opts.lidar_name.upper() == "LIO3T":
            Processed_532(opts.folder, opts.lidar_name ,fn, opts.plot, opts.altitude)
        else:
            Processed_355(opts.folder, opts.lidar_name ,fn, opts.plot,5, opts.altitude)
            Processed_355(opts.folder, opts.lidar_name ,fn, opts.plot,4, opts.altitude)

if __name__ == '__main__':
    main()
