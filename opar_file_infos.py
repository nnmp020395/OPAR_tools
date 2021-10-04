from argparse import Namespace, ArgumentParser
import datetime as dt
import sys 
import numpy as np 
import pandas as pd 
import netCDF4 as nc4
from pathlib import Path
import glob, os

__author__ = "Phuong-N-N NGUYEN"
__email__ = "phuong.nguyen@latmos.ipsl.fr"

"""
Le script retourne les voies, l'instrument du jour sélectionné des données OPAR, 
et retourne des informations de l'altitude, range et la résolution verticale/temporelle. 
Le script indique le nombre des profils dans ce jour, le temps début et le temps fin de mesure du jour. 

Le script a besoin des arguments suivant: 
--folder: le chemin du répertoire parent des données OPAR
--date_start: le jour souhaité yyyy-mm-dd
--date_end: 

"""
def time_from_opar_raw(path):
    data = nc4.Dataset(path, 'r')
    time, calendar, units_time = data.variables['time'][:], data.variables['time'].calendar, data.variables['time'].units
    timeLidar = pd.to_datetime(nc4.num2date(time, units_time, calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True))
    # timeLidar = np.array(timeLidar.strftime("%Y-%m-%d %H:%M")).astype("datetime64[s]")
    return timeLidar 


parser = ArgumentParser()
parser.add_argument("--folder", "-f", type = str, 
    help="Opar path, termined by /", required = True)
parser.add_argument("--date_start", "-start", type = lambda t: dt.datetime.strptime(t, "%Y-%m-%d"), 
    help="Date format = yyyy-mm-dd", required = True)
parser.add_argument("--date_end", "-end", type = lambda t: dt.datetime.strptime(t, "%Y-%m-%d"), 
    help="Date format = yyyy-mm-dd", required = True)
parser.add_argument("--out_path", "-o", type = str, 
    help="Path of output txt file", required = False)
opts = parser.parse_args()
print(opts)

#---------prepare for output text file-------------------
out_path = opts.out_path + "/" + "opar_infos_output.txt"
sys.stdout = open(out_path, "w")
# opar_folder_path = "/home/nmpnguyen/OPAR/"
# date_start = dt.date(2019,1,15)
# date_end = dt.date(2019,1,20)
list_dates = [
    opts.date_start + dt.timedelta(days=d)
    for d in range((opts.date_end - opts.date_start).days + 1)
]

lidar_name = "li1200"
lidar_folder_path = opts.folder+lidar_name.upper()+".daily/"
list_date_files = [os.path.basename(x) for x in glob.glob(lidar_folder_path+"*.nc4")]
list_date_files = np.array(sorted(list_date_files))
numb_date_files = 0
numb_profil = 0
for date in list_dates:
    # files_path = opar_folder_path / f"{date:%Y}" / f"{date:%m}" / f"{date:%d}"
    date_file = list_date_files[list_date_files == f"{date:%Y-%m-%d}"+".nc4"]
    if date_file.size != 0:      
        data = nc4.Dataset(lidar_folder_path+str(date_file[0]), "r")
        numb_profil += data.variables['time'].shape[0]
        numb_date_files += 1

print("-----------------------")
print(f"Instrument: {lidar_name.upper()}")
print(f"Wavelength: 355nm")
print("-----------------------")
print("Number of date files: %d" %numb_date_files)
print("Number of profiles: %d" %numb_profil)

if numb_date_files != 0:
    channel = data.variables['channel'][:]
    for ch,i in zip(channel, range(0, channel.shape[0])):
        print("Voie %d : %s" %(i, ch))
    res = data.variables['range'][1].data - data.variables['range'][0].data
    print("Vertical resolution: %.3f %s" %(res, data.variables['range'].units))
    res = data.variables['time'][1].data - data.variables['time'][0].data
    print("Time resolution: %d %s" %(res, data.variables['time'].units.split(" ")[0]))


lidar_name = "lio3t"
lidar_folder_path = opts.folder+lidar_name.upper()+".daily/"
list_date_files = [os.path.basename(x) for x in glob.glob(lidar_folder_path+"*.nc4")]
list_date_files = np.array(sorted(list_date_files))
numb_date_files = 0
numb_profil = 0
for date in list_dates:
    # files_path = opar_folder_path / f"{date:%Y}" / f"{date:%m}" / f"{date:%d}"
    date_file = list_date_files[list_date_files == f"{date:%Y-%m-%d}"+".nc4"]
    if date_file.size != 0:      
        data = nc4.Dataset(lidar_folder_path+str(date_file[0]), "r")
        numb_profil += data.variables['time'].shape[0]
        numb_date_files += 1

print("-----------------------")
print(f"Instrument: {lidar_name.upper()}")
print(f"Wavelength: 532nm")
print("-----------------------")
print("Number of date files: %d" %numb_date_files)
print("Number of profiles: %d" %numb_profil)

if numb_date_files != 0:
    channel = data.variables['channel'][:]
    for ch,i in zip(channel, range(0, channel.shape[0])):
        print("Voie %d : %s" %(i, ch))
    res = data.variables['range'][1].data - data.variables['range'][0].data
    print("Vertical resolution: %.3f %s" %(res, data.variables['range'].units))
    res = data.variables['time'][1].data - data.variables['time'][0].data
    print("Time resolution: %d %s" %(res, data.variables['time'].units.split(" ")[0]))

sys.stdout.close()