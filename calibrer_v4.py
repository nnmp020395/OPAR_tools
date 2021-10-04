import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from pathlib import Path
import sys
import xarray as xr

import numpy as np
import pandas as pd

"""
  Update 2021-04-07
  Le script réalise les étapes suivantes:
    - range corrected signal and background soustracted
    - peak detection -> low clouds detection 
    - definitiion of height referent 
    - normalization
    - time average
    
   Fontionc init et add sont pour détecter le pic du signal, donc détecter les nuages, aérosols
"""

def init(
    x,
    lag,
    threshold,
    influence,
    ):
    '''
    Smoothed z-score algorithm
    Implementation of algorithm from https://stackoverflow.com/a/22640362/6029703
    '''
    labels = np.zeros(lag)
    filtered_y = np.array(x[0:lag])
    avg_filter = np.zeros(lag)
    std_filter = np.zeros(lag)
    var_filter = np.zeros(lag)
    avg_filter[lag - 1] = np.mean(x[0:lag])
    std_filter[lag - 1] = np.std(x[0:lag])
    var_filter[lag - 1] = np.var(x[0:lag])
    return dict(avg=avg_filter[lag - 1], var=var_filter[lag - 1],
                std=std_filter[lag - 1], filtered_y=filtered_y,
                labels=labels)


def add(result, 
    single_value,
    lag,
    threshold,
    influence,
    ):
    previous_avg = result['avg']
    previous_var = result['var']
    previous_std = result['std']
    filtered_y = result['filtered_y']
    labels = result['labels']
    if abs(single_value - previous_avg) > threshold * previous_std:
        if single_value > previous_avg:
            labels = np.append(labels, 1)
        else:
            labels = np.append(labels, -1)
        # calculate the new filtered element using the influence factor
        filtered_y = np.append(filtered_y, influence * single_value
                               + (1 - influence) * filtered_y[-1])
    else:
        labels = np.append(labels, 0)
        filtered_y = np.append(filtered_y, single_value)
    # update avg as sum of the previuos avg + the lag * (the new calculated item - calculated item at position (i - lag))
    current_avg_filter = previous_avg + 1. / lag * (filtered_y[-1]
            - filtered_y[len(filtered_y) - lag - 1])
    # update variance as the previuos element variance + 1 / lag * new recalculated item - the previous avg -
    current_var_filter = previous_var + 1. / lag * ((filtered_y[-1]
            - previous_avg) ** 2 - (filtered_y[len(filtered_y) - 1
            - lag] - previous_avg) ** 2 - (filtered_y[-1]
            - filtered_y[len(filtered_y) - 1 - lag]) ** 2 / lag)  # the recalculated element at pos (lag) - avg of the previuos - new recalculated element - recalculated element at lag pos ....
    # calculate standard deviation for current element as sqrt (current variance)
    current_std_filter = np.sqrt(current_var_filter)
    return dict(avg=current_avg_filter, var=current_var_filter,
                std=current_std_filter, filtered_y=filtered_y[1:],
                labels=labels)


from argparse import Namespace, ArgumentParser
# def main():
parser = ArgumentParser()
parser.add_argument("--opar_file", "-file", type=str, help="Opar path file", required=True)
parser.add_argument("--ratio", "-ratio", type=float, help="Extinction vs Backscatter ratio coef.", required=True)
parser.add_argument("--wavelengh", "-wave", type=int, help="", required=True)
opts = parser.parse_args()
print(opts)


opar_file = Path(opts.opar_file)#"/home/nmpnguyen/OPAR/LIO3T.daily/2019-01-21.nc4"
wave=opts.wavelengh
d = xr.open_dataset(opar_file)
time = d.time.values
r = d.range.values*1e3
r_square = np.square(r)
alt = r + 2160
rcs = xr.zeros_like(d['signal'])
for c in range(0, len(d["channel"])):
  signal = d["signal"][:,:,c]
  ids = (alt>=80000)&(alt<=100000)
  bck = signal[:,ids].mean(axis=1)        
  rcs[:,:,c] = (signal-bck)*r_square

print('OPAR RANGE CORRECTED DATA-------------------------------end')

alt_ref = 5000
r_id = (alt <= alt_ref)
time_mask = []
id_mask = []
for t in range(time.size):
    if wave == 355:
      profil1 = rcs[t,r_id,5].values #+ rcs[t,r_id,7].values
    else:
      profil1 = rcs[t,r_id,6].values + rcs[t,r_id,7].values
    lag = 5
    threshold = 5
    influence = 0.5
    m=0
    result = init(profil1, lag=lag, threshold=threshold, influence=influence)
    for line in profil1:
        result = add(result, line, lag, threshold, influence)
    if (len(np.where(result['labels']==1)[0]) != 0):
        time_mask.append(time[t])
        id_mask.append(t)


print(f'time index of peak detected: {id_mask}')
if wave == 355:  
  rcs_new = np.array(rcs[:,:,5].values , copy=True)#+ rcs[:,:,7].values
else:
  rcs_new = np.array(rcs[:,:,6].values + rcs[:,:,7].values , copy=True)#
#rcs_new[id_mask,:] = np.nan
rcs_new = pd.DataFrame(rcs_new, index = time, columns=r)
print('PEAK DETECTION-------------------------end')
ratio=str(opts.ratio)
simul_df = pd.read_pickle(list(Path('/homedata/nmpnguyen/OPAR/Processed/', opar_file.parts[4].split('.')[0]).glob(opar_file.name.split('.')[0]+'_'+ratio+'*.pkl'))[0])
mol_varname = 'beta'+str(wave)+'mol'
betamol = simul_df[mol_varname]
betamol = betamol.unstack(level=1)
print('SIMULATED DATA-------------------------------end')
z_cc = (alt > (alt_ref-200))&(alt <= alt_ref)
constk = rcs_new.iloc[:,z_cc].div(betamol.iloc[:,z_cc]).mean(axis=1) 
atb = rcs_new.div(constk, axis=0)
sr = atb.div(betamol)
print('NORMALIZATION-------------------------end')
srAv = sr.resample('30Min').mean()
atbAv = atb.resample('30Min').mean()
betamolAv = betamol.resample('30Min').mean()

idplot = (alt <=20000)
ncols = 4
nrows = srAv.shape[0]%ncols + 1
print(f'cols/rows of plot: {ncols, nrows}')
#fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[16,12])
#for n,ax in enumerate(axs.flatten()):
    #ax.plot(srAv.iloc[n,idplot], alt[idplot], label=str(wave))
    #ax.vlines(1, ymin=alt[0], ymax=alt[idplot][-1], linestyles='--', color='r')
    #ax.set(title=str(srAv.iloc[n,idplot].name))


#plt.suptitle(f'Test of Extinction coef. ratio={ratio}')
#plt.tight_layout()
#plt.savefig(Path('/home/nmpnguyen/test_fig',opar_file.name.split('.')[0]+ratio+'RatioTest.png'))

#nc_id = pd.read_pickle('/homedata/nmpnguyen/OPAR/Processed/LIO3T/2019-01-21_00532.p00532.s_ATB.pkl')
#nc_idAv = nc_id.resample('30Min').mean()

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[16,12])
for n,ax in enumerate(axs.flatten()):
    ax.semilogx(atbAv.iloc[n,idplot], alt[idplot], label=str(wave)+' attn', color='g')
    #ax.plot(np.log(nc_idAv.iloc[n,:]), alt[:nc_idAv.shape[1]], label='532old', color='r')
    ax.semilogx(betamolAv.iloc[n,idplot], alt[idplot], '--', color='k', label='mol attn')
    ax.legend()
    ax.set(title=str(srAv.iloc[n,idplot].name))


plt.suptitle(f'Test of Extinction coef. ratio={ratio}')
plt.tight_layout()
plt.savefig(Path('/home/nmpnguyen/test_fig',opar_file.name.split('.')[0]+ratio+str(wave)+'RatioTest.png'))


#python /home/nmpnguyen/Codes/ear_data_analyse.py -file /home/nmpnguyen/OPAR/LI1200.daily/2019-01-17.nc4 -ratio 0.0595
#python /home/nmpnguyen/Codes/ear_data_analyse.py -file /home/nmpnguyen/OPAR/LIO3T.daily/2019-01-17.nc4 -ratio 0.0595
