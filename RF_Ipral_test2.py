import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import click
import sys,os

# import module "cloud filter"
sys.path.append('/homedata/nmpnguyen/ipral-tools/')
from ipral_chm15k_cloud_filter import ipral_remove_cloud_profiles
# from imp import reload as rl
# rl(ipral_chm15k_cloud_filter)

'''
Le script est pour tester la méthode Rayleigh Fit à un jour de données Ipral.
Appliquer sur tous les profils, non ou ayant average time. 

La méthode de Rayleigh Fit change au niveau de calcul l'épaisseur d'optique. On change les bornes de l'intégration, sépararer l'intégration en fonction de l'altitude référence. 

'''


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
    Cette fonction permet de calculer la retrodiffusion attenuee à partir de zref, data2D
    Input: AlphaMol 2D, Betamol2D, alt 1D, indice de l'altitude médiane entre zmin et zmax
    Output: profil attenue 2D
    '''
    Tr = betamol[:,idxref:].copy() #np.zeros_like(BetaMol[idx_ref:])
    Tr2 = np.zeros_like(betamol[:,idxref:])
    # print(len(Tr), len(Tr2))

    for i,j in zip(range(1, Tr2.shape[1]),range(idxref+1, len(alt))):
        Tr[:,i] = Tr[:,i-1] + alphamol[:,i]*(alt[j]-alt[j-1])
        Tr2[:,i] = np.exp(-2*Tr[:,i])

    betamol_Z0 = betamol.copy() #np.ones_like(BetaMol)   
    betamol_Z0[:,idxref+1:] = betamol[:,idxref+1:]*Tr2[:,1:]
    # print(betamol_Z0[idxref])  
    return betamol_Z0


def get_backscatter_mol_attn_v2(alphamol, betamol, alt):
    '''
    Cette fonction permet de calculer la retrodiffusion attenuee à partir de l'altitude de l'instrument, data2D
    Input: AlphaMol 2D, Betamol2D, alt 1D
    Output: profil attenue 2D
    '''
    Tr = np.zeros_like(betamol)
    Tr2 = np.zeros_like(betamol)
    
    for i in range(1, Tr2.shape[1]):
        Tr[:,i] = Tr[:,i-1] + alphamol[:,i]*(alt[i]-alt[i-1])
        Tr2[:,i] = np.exp(-2*Tr[:,i])
        
    betamol_Z0 = betamol*Tr2        
    return betamol_Z0


def processed(w, pression, ta, ztop, zbottom, data):
    if w == 532:
        channel = 'rcs_17'
    else: 
        channel = 'rcs_13'
    ### Range corrected signal - Background
    signal = data[channel]
    bckgrd = data['bckgrd_'+channel]
    ipralrange = data['range'].values
    iprcs = (signal/(ipralrange**2) - bckgrd)*(ipralrange**2)

    ### Determiner z0
    Z0, idx_ref = get_altitude_reference(zbottom, ztop, ipralrange)
    zintervalle = np.where((ipralrange >= zbottom) & (ipralrange <= ztop))[0]
    '''
    Appliquer aux données Opar : nuage 21.01.2019 et ciel clair 17.06.2019
    1. Calculer le profil Pr2_z0 = Pr2[z]/Pr2[z0] 
    puis calculer la valeur de son intégrale etre zmin et zmax
    '''
    Pr2_Z0 = iprcs/iprcs.isel(range=idx_ref)
    Pr2_integ = np.zeros(len(iprcs['time']))
    for z in zintervalle[:-1]:
        Pr2_integ = Pr2_integ + Pr2_Z0.isel(range=z).values*(ipralrange[z+1]-ipralrange[z])
        
    '''
    2. Calculer le profil BetaMol[z]*Tr2(AlphaMol(z))[z0] 
    Et calculer son integrale entre zmin et zmax
    '''
    AlphaMol, BetaMol = get_backscatter_mol(pression.values, ta.values, w*1e-3)
    # print(BetaMol)
    # BetaMol_Z0_v2 = np.array([get_backscatter_mol_attn_v2(AlphaMol[i,:], BetaMol[i,:], ipralrange) for i in range(len(iprcs['time']))])
    # BetaMol_Z0 = np.array([get_backscatter_mol_attn_v1(AlphaMol[i,:], BetaMol[i,:], ipralrange, idx_ref) for i in range(len(iprcs['time']))])
    BetaMol_Z0 = get_backscatter_mol_attn_v1(AlphaMol, BetaMol, ipralrange, idx_ref)
    # BetaMol_Z0 = get_backscatter_mol_attn_v2(AlphaMol, BetaMol, ipralrange)
    BetaMol_integ = 0
    for z in zintervalle[:-1]:
        BetaMol_integ = BetaMol_integ + BetaMol_Z0[:,z]*(ipralrange[z+1]-ipralrange[z])
        
    '''
    3. Diviser l'un par l'autre pour obtenir cRF
    '''
    cRF = (BetaMol_integ/Pr2_integ).reshape(-1,1)
    '''
    4. Normaliser les profils mesures par cRF
    '''
    Pr2_norm = Pr2_Z0.values*cRF
    return Pr2_norm, BetaMol_Z0



from argparse import Namespace, ArgumentParser  
parser = ArgumentParser()
parser.add_argument("--ztop", "-top", type=int, help="bound top of calibration height", required=True)
parser.add_argument("--zbottom", "-bottom", type=int, help="bound top of calibration height", required=True)
parser.add_argument("--filterHeight", "-fH", type=int, help="Top of Hieght where cloud are filted", required=True)
parser.add_argument("--add_file", "-f", type=str, help="add a file to calibrate", required=False)
parser.add_argument("--out_file", "-o", type=str, help="output suffixe name of add_file", required=False)
opts = parser.parse_args()
print(opts)


import sys
sys.path.append('/homedata/nmpnguyen/ipral-tools/')
# from imp import reload as rl
# rl(ipral_chm15k_cloud_filter, ipral_variables_simulation)
from ipral_chm15k_cloud_filter import ipral_remove_cloud_profiles as cloud_filter
from ipral_variables_simulation import simulate


if opts.add_file:
    ipral_folder = [Path(opts.add_file)]
else:
    IPRAL_PATH = Path("/bdd/SIRTA/pub/basesirta/1a/ipral/2018/")
    ipral_folder = sorted(IPRAL_PATH.glob('**/**/ipral_1a_Lz1R15mF30sPbck_v01_*_000000_1440.nc'))


wavelengths = [532, 355]

### Determiner z0
zbottom = opts.zbottom #5000
ztop = opts.ztop #7000


for file_ipral in ipral_folder: 
    ###____Cloud filter  
    # ipralraw = cloud_filter(date=pd.to_datetime(file_ipral.name.split('_')[4]), alt_max=opts.filterHeight, ipral_file=file_ipral, output=Path('/homedata/nmpnguyen/IPRAL/RF/Cloud_filter/', file_ipral.name))
    print(file_ipral)
    ipralraw = xr.open_dataset(file_ipral)
    ###____Preparation Pression, Temperature et alt 
    # ipralsimul = simulate(Path('/homedata/nmpnguyen/IPRAL/RF/Cloud_filter/', file_ipral.name))
    ipralsimul = simulate(file_ipral)
    # ipralsimulpath = sorted(Path('/homedata/nmpnguyen/IPRAL/RF').glob('ipral_1a_Lz1R15mF30sPbck_v01_20200206*simul.pkl'))
    ipralsimulpath = sorted(Path('/homedata/nmpnguyen/IPRAL/RF/Simul/').glob(file_ipral.name.split('.')[0]+'_simul.pkl'))[0]
    # ipralsimul = pd.read_pickle(ipralsimulpath)
    pression = ipralsimul['pression'].unstack(level=1)
    ta = ipralsimul['ta'].unstack(level=1)
    
    new_signal = np.array([processed(wave, pression, ta, ztop, zbottom, ipralraw)[0] for wave in wavelengths])
    new_simul = np.array([processed(wave, pression, ta, ztop, zbottom, ipralraw)[1] for wave in wavelengths])
    print(new_signal.shape)
    ### Insert output path 
    if opts.out_file:
        output_path = Path("/homedata/nmpnguyen/IPRAL/RF/Calibrated/", file_ipral.name.split('.')[0]+opts.out_file)
    else:
        output_path = Path("/homedata/nmpnguyen/IPRAL/RF/Calibrated/", file_ipral.name)
    
    print(f'output file: {output_path}')
    ### Write output file
    ds = xr.Dataset(data_vars = {"calibrated": (("wavelength","time","range"), new_signal),
                                "simulated": (("wavelength","time","range"), new_simul),}, 
                    coords = {
                        "time":ipralraw['time'].values,
                        "range": ipralraw['range'].values,
                        "wavelength":wavelengths,                        
                    },
                    attrs = {"calibration height": [zbottom, ztop],},) 
    ds.to_netcdf(output_path, 'w')