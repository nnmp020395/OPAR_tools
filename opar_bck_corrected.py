#!/usr/bin/env python3

import os
from pathlib import Path
import shutil
import sys

# import click

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

from argparse import Namespace, ArgumentParser
# def main():
parser = ArgumentParser()
parser.add_argument("--folder", "-f", type=str, help="Main folder of lidar data", required=True)
parser.add_argument("--date", "-d", type=dt.date.fromisoformat(), help = "YYYY-MM-DD daily file")
parser.add_argument("--wavelength", "-w", type=str, help="Name of lidar on upper character", required=True)
parser.add_argument("--output", "-o", type=str, help="Output folder path", required=True)
opts = parser.parse_args()
print(opts)
opar_folder = Path(opts.folder)/Path("LI1200.daily") if opts.wavelength==355 else Path(opts.folder)/Path("LIO3T.daily")
if opts.date is None:
    data_file = sorted(opar_folder.glob("*.nc4"))
    for dt in data_file:
        Processed(dt, Path(opts.output))
else:
    data_file = opar_folder / Path(str(opts.date)+".nc4")
    Processed(data_file, Path(opts.output))


# if __name__ == '__main__':
#     main()

def Processed(data_file, output_path):
    output_file = output_path / data_file.name
    print(f"copying {data_file.name} into {output_file}")
    try:
        new_file = shutil.copy(data_file, output_file)
    except FileNotFoundError:
        print(f"ERROR: no IPRAL file available for {date:%Y-%m-%d}")
        return 1 

    # Copy Opar netcdf file into new file and change rigth on the file
    # new_file = shutil.copy(data_file, output_file)
    os.chmod(new_file, 0o666)
    nc_id = nc4.Dataset(new_file, "a")
    r = nc_id.variables["range"][:]
    alt = r+2.1
    r_square = np.square(r*1e3)
    for c in range(0, len(nc_id.variables["channel"])):
        signal = nc_id.variables["signal"][:,:,c]
        bck = signal[:,np.where((alt>=80)&(alt>=100))[0]].mean(axis=1)
        bck = bck.reshape(bck.shape[0],1)
        new_signal = (signal-bck)*r_square
        nc_id.variables["signal"][:,:,c] = new_signal

    nc_id.variables["signal"].long_name = "range corrected signal background substracted"
    print("end")
    return 0


