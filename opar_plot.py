import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import xarray as xr
from pathlib import Path
import sys, os, glob

import numpy as np
import pandas as pd

from argparse import Namespace, ArgumentParser  
parser = ArgumentParser()
parser.add_argument('--lidar', '-l', type=str, required=True, help='Lidar name')
parser.add_argument('--year', '-yy', type=str, required=False, help='Year')
opts = parser.parse_args()

lidar = opts.lidar
if opts.year:
    opar_folder = sorted(Path(f'/homedata/noel/OPAR/{lidar}.daily/').glob(f'{opts.year}-*.nc4'))
else:
    opar_folder = sorted(Path(f'/homedata/noel/OPAR/{lidar}.daily/').glob(f'*.nc4'))


for oparpath in opar_folder: 
    opar = xr.open_dataset(oparpath)
    limiteZ = np.where(opar['range'][:]<20)[0]

    for i in range(len(opar['channel'])):
        dateStart = pd.to_datetime(oparpath.name.split('.')[0])
        dateEnd = dateStart + pd.DateOffset(1)
        fig, ax=plt.subplots(figsize=[10,6])
        np.log(opar.isel(channel=i, range=limiteZ)['signal']*np.square(opar.isel(range=limiteZ)['range']*1e3)).plot(x='time',y ='range',ax=ax, cmap='viridis', cbar_kwargs={"label": "log(RCS) \nwithout background correction"}, robust=True, ylim=(0, 20), xlim=(dateStart, dateEnd))
        fig.suptitle(oparpath.name)
        plt.tight_layout()
        plt.savefig(f'/homedata/nmpnguyen/OPAR/{lidar}.daily.QL/QL-{opar["channel"][i].values}-{oparpath.name}.png')
        plt.close()