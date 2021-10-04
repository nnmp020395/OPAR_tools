import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from random import randint


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
    Tr = betamol[idxref:].copy() #np.zeros_like(BetaMol[idx_ref:])
    Tr2 = np.zeros_like(betamol[idxref:])

    for i,j in zip(range(1, len(Tr2)),range(idxref+1, len(alt))):
        Tr[i] = Tr[i-1] + alphamol[i]*(alt[j]-alt[j-1])
        Tr2[i] = np.exp(-2*Tr[i])

    betamol_Z0 = betamol.copy() #np.ones_like(BetaMol)   
    betamol_Z0[idxref+1:] = betamol[idxref+1:]*Tr2[1:] 
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


from argparse import Namespace, ArgumentParser  
parser = ArgumentParser()
# parser.add_argument("--clearskylist", "-l", type=str, help="list of clear sky data day", required=True)
parser.add_argument("--ztop", "-top", type=int, help="bound top of calibration height", required=True)
parser.add_argument("--zbottom", "-bottom", type=int, help="bound top of calibration height", required=True)
parser.add_argument("--wavelength", "-w", type=int, help="bound top of calibration height", required=True)
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
    opar_folder = sorted(Path("/home/nmpnguyen/OPAR/LIO3T.daily").glob("2019*.nc4"))
else:
    lidars = "LI1200"
    opar_folder = sorted(Path("/home/nmpnguyen/OPAR/LI1200.daily").glob("2019*.nc4")) 

for oparpath in opar_folder:
    ### Lire et sortir des données brutes: range(km), channel, signal 
    print(oparpath)
    l = oparpath.name.split('.')[0]
    if w*1e-3 == 0.532:
        # print(Path('/home/nmpnguyen/OPAR/LIO3T.daily/', l+'.nc4'))
        # oparpath = Path('/home/nmpnguyen/OPAR/LIO3T.daily/', l+'.nc4')    
        # oparpath = Path('/home/nmpnguyen/OPAR/LIO3T.daily/2019-01-16.nc4')
        opar = xr.open_dataset(oparpath)
        opalt = opar['range'].values*1e3 + 2160
        oparraw = opar.sel(channel='00532.p')#.isel(time=100)
        oparraw2 = opar.sel(channel='00532.p')+opar.sel(channel='00532.s')
    else:
        # print(Path('/home/nmpnguyen/OPAR/LI1200.daily/', l+'.nc4'))
        # oparpath = Path('/home/nmpnguyen/OPAR/LI1200.daily/', l+'.nc4')    
        # oparpath = Path('/home/nmpnguyen/OPAR/LI1200.daily/2019-01-16.nc4')
        opar = xr.open_dataset(oparpath)
        opalt = opar['range'].values*1e3 + 2160
        oparraw = opar.sel(channel='00355.o.Verylow')
        oparraw2 = opar.sel(channel='00355.o.Low')
    ### Range corrected signal 
    rep1 = range_corrected_signal(oparraw, opar['range'].values*1e3, opar['range'].values*1e3+2160, True)
    rep2 = range_corrected_signal(oparraw2, opar['range'].values*1e3, opar['range'].values*1e3+2160, True)
    ### Lire et sortir la Pression et la Temperature  
    if w*1e-3 == 0.532:
        oparsimulpath = Path('/homedata/nmpnguyen/OPAR/Processed/LIO3T/', l+'_simul.pkl')
    else:
        oparsimulpath = Path('/homedata/nmpnguyen/OPAR/Processed/LI1200/', l+'_simul.pkl')
        
    oparsimul = pd.read_pickle(oparsimulpath)
    pression = oparsimul['pression'].unstack(level=1)
    ta = oparsimul['ta'].unstack(level=1)
    opalt = opalt[:pression.shape[1]] 
    ### Determiner z0
    zbottom = opts.zbottom #5000
    ztop = opts.ztop #7000
    Z0, idx_ref = get_altitude_reference(zbottom, ztop, opalt)
    zintervalle = np.where((opalt >= zbottom) & (opalt <= ztop))[0]
    '''
    Appliquer aux données Opar : nuage 21.01.2019 et ciel clair 17.06.2019
    1. Calculer le profil Pr2_z0 = Pr2[z]/Pr2[z0] 
    puis calculer la valeur de son intégrale etre zmin et zmax
    '''
    Pr2_Z0 = rep2/rep2.isel(range=idx_ref)
    Pr2_integ = np.zeros(len(rep2['time']))
    for z in zintervalle[:-1]:
        Pr2_integ = Pr2_integ + Pr2_Z0.isel(range=z)['signal'].values*(opalt[z+1]-opalt[z])
    
    '''
    2. Calculer le profil BetaMol[z]*Tr2(AlphaMol(z))[z0] 
    Et calculer son integrale entre zmin et zmax
    '''
    AlphaMol, BetaMol = get_backscatter_mol(pression.values, ta.values, w*1e-3) 
    BetaMol_Z0_v2 = np.array([get_backscatter_mol_attn_v2(AlphaMol[i,:], BetaMol[i,:], opalt) for i in range(len(rep2['time']))])
    BetaMol_Z0 = np.array([get_backscatter_mol_attn_v1(AlphaMol[i,:], BetaMol[i,:], opalt, idx_ref) for i in range(len(rep2['time']))])

    BetaMol_integ = 0
    BetaMol_integ_v2 = 0
    for z in zintervalle[:-1]:
        BetaMol_integ = BetaMol_integ + BetaMol_Z0[:,z]*(opalt[z+1]-opalt[z]) 
        BetaMol_integ_v2 = BetaMol_integ_v2 + BetaMol_Z0_v2[:,z]*(opalt[z+1]-opalt[z]) 

    '''
    3. Diviser l'un par l'autre pour obtenir cRF
    '''
    cRF = (BetaMol_integ/Pr2_integ).reshape(-1,1)
    cRF_v2 = (BetaMol_integ_v2/Pr2_integ).reshape(-1,1)
    '''
    4. Normaliser les profils mesures par cRF
    '''
    Pr2_norm = Pr2_Z0['signal'].values*cRF
    Pr2_norm_v2 = Pr2_Z0['signal'].values*cRF_v2

    '''
    5. Calculer le résidus residus et l'intégrale sur l'altitude
    '''
    if w*1e-3 == 0.532:
        lidars = 'LIO3T'
    else:
        lidars = 'LI1200'
        
    residus = np.zeros(len(rep2['time']))
    for t in range(1, len(opalt)):
        residus = residus + np.nansum([Pr2_norm[:,t], -BetaMol_Z0[:,t]], axis=0)#*(opalt[z]-opalt[t-1])

    fichier = open(Path('/homedata/nmpnguyen/OPAR/Processed/RF/', lidars, l, l+'_residus.txt'), 'w')
    fichier.write(f'calibration altitude: {[zbottom, ztop]}')
    fichier.write(f'wavelength: {w}')
    for i in range(len(residus)):
        fichier.write(f'\n{round(residus[i],6)},{str(rep2.time[i].values)}')

    fichier.close()
    '''
    6. Plot un profil normalisé et moléculaire
    '''
    for n in range(0, len(rep2['time']), 20):
        plt.clf()
        fig, (ax3, ax4) = plt.subplots(nrows=1, ncols=2, figsize=[12,6], sharey=True)#
        ax3.semilogx(Pr2_norm[n,:len(opalt)], opalt, label='signal normalisé', color='b')
        ax3.semilogx(BetaMol_Z0[n,:], opalt, label='signal moléculaire attn depuis z0', color='orange')
        ax3.semilogx(BetaMol_Z0_v2[n,:], opalt, label='signal moleculaire attn depuis sol', color='g')
        ax3.axhspan(zbottom, ztop, color='y', alpha=0.5, lw=0, label='calibration height')
        ax3.set_ylim(0,20000)
        # ax.set(title='5.916e-11')
        leg3 = ax3.legend()
        leg3.set_title(f'4/ cRF = {cRF[n]}')
        ax3.set(xlabel='backscatter, 1/m.sr')
        # ax3.set_xlim(1e-8, 2e-6)
        ax4.plot(Pr2_norm_v2[n,:len(opalt)]/BetaMol_Z0_v2[n,:], opalt, label='signal SR, depuis sol', color='g')
        ax4.plot(Pr2_norm[n,:len(opalt)]/BetaMol_Z0[n,:], opalt, label='signal SR, depuis z0', color='orange')
        ax4.axhspan(zbottom, ztop, color='y', alpha=0.5, lw=0, label='calibration height')
        ax4.vlines(1, ymin=opalt.min(), ymax=opalt.max(), color='k', zorder=10)
        # ax.set(title='5.916e-11')
        leg4 = ax4.legend()
        leg4.set_title(f'4/ cRF = {cRF[n]}')
        ax4.set_xlim(-0.5, 3.5)
        ax4.set(xlabel = 'scattering ratio')
        plt.suptitle(f'zmin = {zbottom}, zmax = {ztop}, z0 = {opalt[idx_ref]}\n {lidars}: {str(rep2.time[n].values)}')
        print(Path('/homedata/nmpnguyen/OPAR/Processed/RF/', lidars, l, str(rep2.time[n].values)+'RF_v1.png'))
        plt.savefig(Path('/homedata/nmpnguyen/OPAR/Processed/RF/', lidars, l, str(rep2.time[n].values)+'RF_v1.png'))
        plt.close(fig)
    '''
    7. Save file 
    '''
    