#!/usr/bin/env python3

import os
from pathlib import Path
import shutil
import sys

# import click
import xarray as xr
import netCDF4 as nc4
import numpy as np
import datetime as dt

__author__ = "N-M-Phuong Nguyen"
__version__ = "0.1.0"

"""
Script is used to make new file of range corrected signal background substracted

Input: 
--folder: folder path of opar data
--wavelength: (integer) wavelength in nm (355nm or 532nm)
--date: yyyy-mm-dd date file selected, or None if you want convert all files

Output: NetCDF output file at the same name with Input file, 
records range corrected signal background substracted and 
keep the same variables name. 
-- output_folder: output folder path.

"""
def Processed(opar_file, output_path):

    output_file = output_path / Path(opar_file.name.split(".")[0]+".nc")
    #print(f"copying {opar_file.name} into {output_file}")
    #try:
        #new_file = shutil.copy(opar_file, output_file)
    #except FileNotFoundError:
        #print(f"ERROR: no OPAR file available for {opar_file.name}")
        #return 1 

    # Copy Opar netcdf file into new file and change rigth on the file
    # new_file = shutil.copy(opar_file, output_file)
    # os.chmod(new_file, 0o666)
    nc_id = xr.open_dataset(opar_file)
    r = nc_id['range']
    alt = r+2.160
    r_square = np.square(r*1e3)
    rcs = xr.zeros_like(nc_id['signal'])
    for c in range(0, len(nc_id["channel"])):
      signal = nc_id["signal"][:,:,c]
      ids = (alt>=80)&(alt>=100)
      bck = signal[:,ids].mean(axis=1)        
      rcs[:,:,c] = (signal-bck)*r_square
      rcs[:,:,c].attrs['long_name'] = "range corrected signal background substracted"
	
    nc_id.assign(rcs = rcs)
    print(nc_id)
    nc_id.to_netcdf(output_file)
    print('------------Create new netcdf--------------')
    print(f'-----------Check new netcdf------------\n {xr.open_dataset(output_file)}')
    print("end")
    return 0


from argparse import Namespace, ArgumentParser
# def main():
parser = ArgumentParser()
parser.add_argument("--folder", "-f", type=str, help="Main folder of lidar data", required=True)
# parser.add_argument("--date", "-d", type=dt.date.fromisoformat(), help = "YYYY-MM-DD daily file")
parser.add_argument("--date", "-d", type = str, help = "Input format: YYYY-MM-DD")
parser.add_argument("--wavelength", "-w", type=int, help="Name of lidar on upper character", required=True)
parser.add_argument("--output", "-o", type=str, help="Output folder path", required=True)
opts = parser.parse_args()
print(opts)
opar_folder = Path(opts.folder)/Path("LI1200.daily") if opts.wavelength==355 else Path(opts.folder)/Path("LIO3T.daily")
if opts.date is None:
    opar_file = sorted(opar_folder.glob("*.nc4"))
    for dt in opar_file:
        Processed(dt, Path(opts.output))
else:
    opar_file = opar_folder / Path(opts.date+".nc4")
    Processed(opar_file, Path(opts.output))


# if __name__ == '__main__':
#     main()




