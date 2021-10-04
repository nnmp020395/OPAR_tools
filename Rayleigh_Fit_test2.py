import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from random import randint
import pickle

def range_corrected_signal(signal_raw, opar_range, opar_alt, bck_correction = False):
    '''
    Fontionc permet de retrouver un signal corrigé de la distance à l'instrument et du fond ciel 
    Input: 
        signal_raw: raw signal (MHz) without any correction
        opar_range: range in meters 
        opar_alt: altitude in meters
        bck_correction: False = non corriger, True = corriger
    Output:
        Signal corrigé 
    '''
    if bck_correction == False:
        rcs = signal_raw * (opar_range**2) #MHz.m^2
    else:
        idx = ((opar_alt>=80000)&(opar_alt<=100000))
        signal_bck = signal_raw.isel(range=idx)
        bck = signal_bck.mean(("range"))
        rcs = (signal_raw - bck)*(opar_range**2)
    return rcs


def get_altitude_reference(zbottom, ztop, altitude_total):
    '''
    Fonction permet de retrouver la position de l'altitude référence dans le vecteur de l'altitude et la valeur de altitude référence.
    Input:
        zbottom: le borne bas de l'intervale
        ztop: le borne haut de l'intervale
        altitude_total: le vecteur total de l'altitude
    Output:
        la valeur de l'altitude et son indexe dans le vecteur total
    '''
    def arg_median(a):
        '''
        Fonction permet de retrouver la position médiane de la zone de référence de l'altitude
        Input: 
            a = l'intervale de l'altitude où se trouve la zone de référence 
        Ouput:
            Indexe de la position dans cet intervale
        '''
        if len(a) % 2 == 1:
            return np.where(a == np.median(a))[0][0]
        else:
            l,r = len(a) // 2 - 1, len(a) // 2
            left = np.partition(a, l)[l]
            right = np.partition(a, r)[r]
            return np.where(a == left)[0][0]

    interval_ref = altitude_total[(altitude_total >= zbottom) & (altitude_total <= ztop)] 
    idxref = arg_median(interval_ref)
    zref = interval_ref[idxref]
    idxref_in_total = np.where(altitude_total == zref)[0][0]
    return zref, idxref_in_total

'''
2. Calculer le profil BetaMol[z]*Tr2(AlphaMol(z))[z0] 
Et calculer son integrale entre zmin et zmax
'''
def get_backscatter_mol(p, T, w):
    '''
    Fonction permet de calculer le coef. de backscatter moléculaire 
    p(Pa), T(K), w(um)
    '''
    k = 1.38e-23
    betamol = (p/(k*T) * 5.45e-32 * (w/0.55)**(-4.09))
    alphamol = betamol/0.119
    return alphamol, betamol


def get_backscatter_mol_attn_v1(alphamol, betamol, alt, idxref):
    '''
    Cette fonction permet de calculer la retrodiffusion attenuee à partir de zref 
    '''
    Tr = betamol[:,idxref:].copy() #np.zeros_like(BetaMol[idx_ref:])
    Tr2 = np.zeros_like(betamol[:,idxref:])

    for i,j in zip(range(1, Tr2.shape[1]),range(idxref+1, len(alt))):
        Tr[:,i] = Tr[:,i-1] + alphamol[:,i]*(alt[j]-alt[j-1])
        Tr2[:,i] = np.exp(-2*Tr[:,i].astype(float))

    betamol_Z0 = betamol.copy() #np.ones_like(BetaMol)   
    betamol_Z0[:,idxref+1:] = betamol[:,idxref+1:]*Tr2[:,1:] 
    return betamol_Z0


def get_backscatter_mol_attn_v2(alphamol, betamol, alt):
    '''
    Cette fonction permet de calculer la retrodiffusion attenuee à partir de l'altitude de l'instrument
    '''
    Tr = np.zeros_like(betamol)
    Tr2 = np.zeros_like(betamol)
    
    for i in range(1, len(Tr)):
        Tr[i] = Tr[i-1] + alphamol[i]*(alt[i]-alt[i-1])
        Tr2[i] = np.exp(-2*Tr[i])
        
    betamol_Z0 = betamol*Tr2        
    return betamol_Z0



def processed(w, channel, pression, ta, ztop, zbottom, data):
    oparraw = data.sel(channel=channel)
    opalt = data['range'].values*1e3 + 2160
    ### Range corrected signal 
    rep1 = range_corrected_signal(oparraw, data['range'].values*1e3, data['range'].values*1e3+2160, True)
    ### Determiner z0
    Z0, idx_ref = get_altitude_reference(zbottom, ztop, opalt)
    zintervalle = np.where((opalt >= zbottom) & (opalt <= ztop))[0]
    '''
    Appliquer aux données Opar : nuage 21.01.2019 et ciel clair 17.06.2019
    1. Calculer le profil Pr2_z0 = Pr2[z]/Pr2[z0] 
    puis calculer la valeur de son intégrale etre zmin et zmax
    '''
    Pr2_Z0 = rep1/rep1.isel(range=idx_ref)
    Pr2_integ = np.zeros(len(rep1['time']))
    for z in zintervalle[:-1]:
        Pr2_integ = Pr2_integ + Pr2_Z0.isel(range=z)['signal'].values*(opalt[z+1]-opalt[z])
    
    '''
    2. Calculer le profil BetaMol[z]*Tr2(AlphaMol(z))[z0] 
    Et calculer son integrale entre zmin et zmax
    '''
    AlphaMol, BetaMol = get_backscatter_mol(pression.values, ta.values, w*1e-3) 
    # BetaMol_Z0_v2 = np.array([get_backscatter_mol_attn_v2(AlphaMol[i,:], BetaMol[i,:], opalt) for i in range(len(rep1['time']))])
    BetaMol_Z0 = get_backscatter_mol_attn_v1(AlphaMol, BetaMol, opalt, idx_ref)
    BetaMol_integ = 0
    BetaMol_integ_v2 = 0
    for z in zintervalle[:-1]:
        BetaMol_integ = BetaMol_integ + BetaMol_Z0[:,z]*(opalt[z+1]-opalt[z]) 
        # BetaMol_integ_v2 = BetaMol_integ_v2 + BetaMol_Z0_v2[:,z]*(opalt[z+1]-opalt[z]) 

    '''
    3. Diviser l'un par l'autre pour obtenir cRF
    '''
    cRF = (BetaMol_integ/Pr2_integ).reshape(-1,1)
    # cRF_v2 = (BetaMol_integ_v2/Pr2_integ).reshape(-1,1)
    '''
    4. Normaliser les profils mesures par cRF
    '''
    Pr2_norm = Pr2_Z0['signal'].values*cRF
    # Pr2_norm_v2 = Pr2_Z0['signal'].values*cRF_v2    
    return Pr2_norm, BetaMol_Z0


'''
Import dataset
'''
from argparse import Namespace, ArgumentParser  
parser = ArgumentParser()
# parser.add_argument("--clearskylist", "-l", type=str, help="list of clear sky data day", required=True)
parser.add_argument("--ztop", "-top", type=int, help="bound top of calibration height", required=True)
parser.add_argument("--zbottom", "-bottom", type=int, help="bound top of calibration height", required=True)
parser.add_argument("--wavelength", "-w", type=int, help="bound top of calibration height", required=True)
parser.add_argument("--add_file", "-f", type=str, help="add a file to calibrate", required=False)
parser.add_argument("--out_file", "-o", type=str, help="output suffixe name of add_file", required=False)
opts = parser.parse_args()
print(opts)

### Ouvrir le fichier des jours sélectionés
# with open(opts.clearskylist, 'r') as f: #'/home/nmpnguyen/Codes/ClearSkyLIO3Tlist'
#     all_data = [line.strip() for line in f.readlines()]
    
# metadata_line = all_data[:4]
# listdays = all_data[4:]
# w = int(all_data[3].split(': ')[1])

w = opts.wavelength
if w*1e-3 == 0.532:
    lidars = "LIO3T"
    channels = ['00532.p', '00532.s']
    # opar_folder = sorted(Path("/home/nmpnguyen/OPAR/LIO3T.daily").glob("2019*.nc4"))
else:
    lidars = "LI1200"
    channels = ['00355.o.Verylow', '00355.o.Low']
    # opar_folder = sorted(Path("/home/nmpnguyen/OPAR/LI1200.daily").glob("2019*.nc4")) 

if opts.add_file:
    opar_folder = [Path(opts.add_file)]
else:
    opar_folder = sorted(Path("/homedata/noel/OPAR/", lidars+".daily").glob("2018*.nc4"))

    
for oparpath in opar_folder:
    print(oparpath)
    l = oparpath.name.split('.')[0]
    ### Lire et sortir des données brutes: range(km), channel, signal   
    opar = xr.open_dataset(oparpath)
    oparsimulpath = Path('/homedata/nmpnguyen/OPAR/Processed/', lidars, l+'_simul.pkl')        
    oparsimul = pd.read_pickle(oparsimulpath)
    pression = oparsimul['pression'].unstack(level=1)
    ta = oparsimul['ta'].unstack(level=1)

    ### Determiner z0
    zbottom = opts.zbottom #5000
    ztop = opts.ztop #7000
    '''
    7. Sortir en NetCDF
    '''
    new_signal = np.array([processed(w, ch, pression, ta, ztop, zbottom, opar)[0] for ch in channels])
    new_simul = np.array([processed(w, ch, pression, ta, ztop, zbottom, opar)[1] for ch in channels])
    print(new_signal.shape)
    ### Insert output path 
    if opts.out_file:
        output_path = Path("/homedata/nmpnguyen/OPAR/Processed/RF/", lidars, l+opts.out_file)
        opt_write = 'w'
    else:
        output_path = Path("/homedata/nmpnguyen/OPAR/Processed/RF/", lidars, l+"RF_v1.nc")
        opt_write = 'a'
    
    print(f'output file: {output_path}')
    ### Write output file
    ds = xr.Dataset(data_vars = {"calibrated": (("channel","time","range"), new_signal),
                                "simulated": (("channel","time","range"), new_simul),}, 
                    coords = {
                        "time": opar['time'].values,
                        "range": opar['range'].values,
                        "channel": channels,                        
                    },
                   attrs = {"calibration height": [zbottom, ztop],},)  
    ds.to_netcdf(output_path, opt_write)
    # picklefilename = Path('/homedata/nmpnguyen/OPAR/Processed/RF/', lidars, l+'RF_v1.pkl')
    # open_file = open(picklefilename, "wb")
    # pickle.dump(df, open_file)
    # open_file.close()