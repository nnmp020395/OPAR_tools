from argparse import Namespace, ArgumentParser
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
from os.path import abspath
from matplotlib import colors


"""
Le script sert à plotter les données raw du ficher sélectionné. 

__author__ = "N-M-Phuong Nguyen"
__email__ = "marc-antoine.drouin@lmd.ipsl.fr"
__version__ = "1.0.0"

"""

def time_from_opar_raw(path):
    """
    Read and convert decimal format to datetime format of time variable.
    path = daily file path
    """
    data = nc4.Dataset(path, 'r')
    time, calendar, units_time = data.variables['time'][:], data.variables['time'].calendar, data.variables['time'].units
    timeLidar = pd.to_datetime(nc4.num2date(time, units_time, calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True))
    # timeLidar = np.array(timeLidar.strftime("%Y-%m-%d %H:%M")).astype("datetime64[s]")
    return timeLidar 


def get_path_out(main_path, filename, time_index, lidar_name, channel, arranging = False):
    """
    Créer le chemin de du fichier résultat

    """
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
    return path_file_out




def quicklook_355_plot(li1200, channel_numb):
    def get_signal_significatif(signal):
        array_nozero = np.where(signal)
        if len(array_nozero[0])/(signal.shape[0]*signal.shape[1]) > 0.1:
            return signal
        else:
            return None 
    dt1200 = nc4.Dataset(li1200, 'r')
    t1200 = time_from_opar_raw(li1200)
    rangeLi = dt1200.variables['range'][:][np.where(dt1200.variables['range'][:]<=100)]
    altLi = rangeLi + 2.1
    channel = dt1200.variables['channel'][channel_numb]
    print(channel)
    signal_before = (dt1200.variables['signal'][:,:rangeLi.shape[0],channel_numb])
    fc = dt1200.variables['signal'][:,np.where(altLi>80)[0],channel_numb]
    mean_fc = fc.mean(axis =1).reshape(-1, 1)
    signal_li1200 = signal_before - mean_fc
    if get_signal_significatif(signal_li1200) is None:
        print("Signal is None")
    else:
        plt.cla()
        plt.clf()
        fig, ax = plt.subplots()
        fmt = plt_dates.DateFormatter("%H:%M")
        p0=ax.pcolormesh(t1200, altLi, signal_li1200.T, shading = "nearest", norm = colors.LogNorm())
        ax.set_ylim(0.0, 25.0)
        ax.xaxis.set_major_formatter(fmt)
        plt.setp(ax.get_xticklabels(), rotation = 90)
        colorbar0 = plt.colorbar(p0, ax=ax)
        fig.suptitle(li1200+"\n Quicklook product level 0-" + channel)
        plt.tight_layout()
        plt.savefig(get_path_out(main_path="/homedata/nmpnguyen/OPAR/Fig/", filename=li1200, time_index=t1200[0], lidar_name="LI1200", channel= channel+"_pcolor_l0", arranging=""))
        plt.close(fig)
        # plt.savefig("/homedata/nmpnguyen/OPAR/Fig/"+fn"_LI1200_raw1.png")
        signal_li1200 = pd.DataFrame(signal_li1200, index = t1200)
        rangeLi = rangeLi*1000
        print(rangeLi)
        for t in t1200:
            plt.cla()
            plt.clf()
            fg, ax = plt.subplots()
            ax.semilogx(signal_li1200.loc[t]*(rangeLi**2), altLi, label = "range^2 correction")
            ax.semilogx(signal_li1200.loc[t], altLi, label = "sky background correction")
            ax.legend()
            ax.set_ylim(0.0, 25.0)
            ax.set(xlabel="signal", ylabel = "Alt,km")
            plt.tight_layout()
            plt.savefig(get_path_out(main_path="/homedata/nmpnguyen/OPAR/Fig", filename=li1200, time_index=t, 
                lidar_name = "LI1200", channel = channel+"_l1", arranging="day"))
            plt.close(fg)

i=0
for f in glob.glob("/home/nmpnguyen/OPAR/LI1200.daily/*.nc4"):
    i+=1
    print(i,f)
    if i>=49:
        quicklook_355_plot(f, 4)
        quicklook_355_plot(f, 5)


def quicklook_532_plot(lio3t):
    def get_signal_significatif(signal):
        array_nozero = np.where(signal)
        if len(array_nozero[0])/(signal.shape[0]*signal.shape[1]) > 0.1:
            return signal
        else:
            return None 
    dtlio3t = nc4.Dataset(lio3t, 'r')
    tlio3t = time_from_opar_raw(lio3t)
    rangeLi = dtlio3t.variables['range'][:][np.where(dtlio3t.variables['range'][:]<=100)]
    altLi = rangeLi + 2.1
    signal_before = dtlio3t.variables['signal'][:,:rangeLi.shape[0],6]+dtlio3t.variables['signal'][:,:rangeLi.shape[0],7]
    fc = dtlio3t.variables['signal'][:,np.where(altLi>80)[0],6] + dtlio3t.variables['signal'][:,np.where(altLi>80)[0],7]
    mean_fc = fc.mean(axis =1).reshape(-1, 1)
    signal_lio3t = pd.DataFrame(signal_before - mean_fc)
    channel = dtlio3t.variables['channel'][6] + dtlio3t.variables['channel'][7]
    if get_signal_significatif(signal_lio3t) is None:
        print("Signal is None")
    else:
        plt.cla()
        plt.clf()
        fig, ax = plt.subplots()
        fmt = plt_dates.DateFormatter("%H:%M")
        p0=ax.pcolormesh(tlio3t, altLi, signal_lio3t.T, shading = "nearest", norm = colors.LogNorm())
        ax.set(ylabel="Alt, km", xlabel="Time")
        ax.set_ylim(0.0, 25.0)
        plt.setp(ax.get_xticklabels(), rotation = 90)
        colorbar0 = plt.colorbar(p0, ax=ax)
        fig.suptitle(lio3t +"\n Quicklook product level 0-" + channel)
        plt.tight_layout()
        plt.savefig(get_path_out(main_path="/homedata/nmpnguyen/OPAR/Fig/", filename=lio3t, time_index=tlio3t[0], lidar_name="LIO3T", channel= channel+"_pcolor_l0", arranging=""))
        plt.close(fig)
        # plt.savefig("/homedata/nmpnguyen/OPAR/Fig/"+fn"_LIO3T_raw1.png")
        signal_lio3t = pd.DataFrame(signal_lio3t, index = tlio3t)
        rangeLi = rangeLi*1000
        print(rangeLi)
        for t in tlio3t:
            plt.cla()
            plt.clf()
            fg, ax = plt.subplots()
            ax.semilogx(signal_lio3t.loc[t]*(rangeLi**2), altLi, label = "range^2 correction")
            ax.semilogx(signal_lio3t.loc[t], altLi, label = "sky background correction")
            ax.legend()
            ax.set_ylim(0.0, 25.0)
            ax.set(xlabel="signal", ylabel = "Alt,km")
            plt.tight_layout()
            plt.savefig(get_path_out(main_path="/homedata/nmpnguyen/OPAR/Fig", filename=lio3t, time_index=t, 
                lidar_name = "LIO3T", channel = channel+"_l1", arranging="day"))
            plt.close(fg)


i=0
for f in glob.glob("/home/nmpnguyen/OPAR/LIO3T.daily/*.nc4"):
    i+=1
    print(i,f)
    quicklook_532_plot(f)

def profils_oneplot(fn):
    li1200 = "/home/nmpnguyen/OPAR/LI1200.daily/"+fn+".nc4"
    dt1200 = nc4.Dataset(li1200, 'r')
    lio3t = "/home/nmpnguyen/OPAR/LIO3T.daily/"+fn+".nc4"
    dtlio3t = nc4.Dataset(lio3t, 'r')
    t1200 = time_from_opar_raw(li1200)
    tlio3t = time_from_opar_raw(lio3t)
    print(dt1200.variables['channel'][:])
    print(dtlio3t.variables['channel'][:])
    range_li1200 = dt1200.variables['range'][:][np.where(dt1200.variables['range'][:]<=100)]
    range_lio3t = dtlio3t.variables['range'][:][np.where(dtlio3t.variables['range'][:]<=100)]
    signal_li1200VL = dt1200.variables['signal'][:,:range_li1200.shape[0],4]
    signal_li1200L = dt1200.variables['signal'][:,:range_li1200.shape[0],5]
    signal_li1200H = dt1200.variables['signal'][:,:range_li1200.shape[0],3]
    signal_li1200M = dt1200.variables['signal'][:,:range_li1200.shape[0],2]

    signal_lio3tPS = dtlio3t.variables['signal'][:,:range_lio3t.shape[0],6]+dtlio3t.variables['signal'][:,:range_lio3t.shape[0],7]
    signal_lio3tP = dtlio3t.variables['signal'][:,:range_lio3t.shape[0],6]
    signal_lio3t0BT = dtlio3t.variables['signal'][:,:range_lio3t.shape[0],4]
    signal_lio3t0BC = dtlio3t.variables['signal'][:,:range_lio3t.shape[0],5]
    f, ax=plt.subplots(2,2, sharey=True, sharex = True, figsize=(9, 9))
    ax[0,0].semilogx(signal_li1200VL[10,:], range_li1200+2.1, label="355 raw,VL")
    ax[0,0].semilogx(signal_li1200L[10,:], range_li1200+2.1, label="355 raw,L")
    ax[0,0].semilogx(signal_li1200M[10,:], range_li1200+2.1, label="355 raw,M")
    ax[0,0].semilogx(signal_li1200H[10,:], range_li1200+2.1, label="355 raw,H")
    # ax[0].semilogx(signal_lio3t[5,:], range_lio3t+2.1, color="green", label="532 raw,p+s")
    ax[0,0].legend()
    ax[0,0].set(xlabel="signal", ylabel = "alt, km")
    ax[0,0].set_ylim(0, 25)
    ax[0,1].semilogx(signal_li1200VL[10,:]*(range_li1200**2), range_li1200+2.1, label="355 raw,VL")
    ax[0,1].semilogx(signal_li1200L[10,:]*(range_li1200**2), range_li1200+2.1, label="355 raw,L")
    ax[0,1].semilogx(signal_li1200M[10,:]*(range_li1200**2), range_li1200+2.1, label="355 raw,M")
    ax[0,1].semilogx(signal_li1200H[10,:]*(range_li1200**2), range_li1200+2.1, label="355 raw,H")
    # ax[1].semilogx(signal_lio3tPS[5,:]*(range_lio3t**2), range_lio3t+2.1, color="green", label="532 raw,p+s")
    ax[0,1].legend()
    ax[0,1].set(xlabel="signal x range²", ylabel = "alt, km")

    ax[1,0].semilogx(signal_lio3tPS[5,:], range_lio3t+2.1, label="532 raw,p+s")
    ax[1,0].semilogx(signal_lio3tP[5,:], range_lio3t+2.1, label="532 raw,p")
    ax[1,0].semilogx(signal_lio3t0BT[5,:], range_lio3t+2.1, label="532 raw,oBT")
    ax[1,0].semilogx(signal_lio3t0BC[5,:], range_lio3t+2.1, label="532 raw,oBC")
    ax[1,0].legend()

    ax[1,1].semilogx(signal_lio3tPS[5,:]*(range_lio3t**2), range_lio3t+2.1, label="532 raw,p+s")
    ax[1,1].semilogx(signal_lio3tP[5,:]*(range_lio3t**2), range_lio3t+2.1, label="532 raw,p")
    ax[1,1].semilogx(signal_lio3t0BT[5,:]*(range_lio3t**2), range_lio3t+2.1, label="532 raw,oBT")
    ax[1,1].semilogx(signal_lio3t0BC[5,:]*(range_lio3t**2), range_lio3t+2.1, label="532 raw,oBC")
    ax[1,1].legend()
    plt.suptitle(fn + str(t1200[10]))
    plt.tight_layout()
    plt.savefig("/homedata/nmpnguyen/OPAR/Fig/"+fn+"alti"+".png")



from argparse import Namespace, ArgumentParser
def main():
    parser = ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, help="Main folder of lidar data")
    parser.add_argument("--day", "-d", type=str, help = "YYYY-MM-DD daily file")
    parser.add_argument("--wavelength", "-w", type=int, help="Name of lidar on upper character", required=True)
    parser.add_argument("--channel_numb", "-c", type=int, help="Number of channel")
    opts = parser.parse_args()
    args = Namespace()
    print(opts)
    if opts.folder is not None:
        if opts.wavelength == 532:
            os.chdir(opts.folder +"/LIO3T.daily/")
            for fn in glob.glob("*.nc4"):
                quicklook_532_plot(fn)
        else:
            os.chdir(opts.folder +"/LI1200.daily/")
            for fn in glob.glob("*.nc4"):
                quicklook_355_plot(fn, opts.channel_numb)
    else:
        fn = opts.day
        if opts.wavelength == 532:
            quicklook_532_plot(fn, opts.channel_numb)
        else
            quicklook_355_plot(fn, opts.channel_numb)

if __name__ == '__main__':
    main()

