import numpy as np 
import netCDF4 as nc4
from netCDF4 import Dataset
import glob, os

"""
Le script sert à enlever les fichiers d'erreur de OPAR. 
Le script demande 
--opar_folder_path : le chemin du répertoire parental des données OPAR
--lidar_name : le nom de Lidar
--channel_numb : index de channel affiché par opar_file_infos.py 
"""

def remove_error_file(opar_folder_path, lidar_name, channel_numb):
    # os.chdir("/home/nmpnguyen/OPAR/LI1200.daily/")
    opar_folder_path = opar_folder_path+lidar_name.upper()+".daily/"
    os.chdir(opar_folder_path)
    i=0
    list_files = np.array(glob.glob("2019*.nc4"))
    mask_profil = []
    for fn in list_files:
        i+=1
        print(i)
        data = nc4.Dataset(opar_folder_path+fn)
        rangeLi = data.variables['range'][:].data
        signal = data.variables['signal'][:,:np.where(rangeLi<=100)[0].shape[0],channel_numb].data
        array_nozero = np.where(signal)
        if len(array_nozero[0])/(signal.shape[0]*signal.shape[1]) > 0.1:
            print(f"file selected: {fn}")
        else:
            print(f"file removed: {fn}")
            list_files = list_files[list_files != fn]
            mask_profil = np.append(mask_profil, fn)
    return list_files
    