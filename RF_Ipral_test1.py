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
    Cette fonction permet de calculer la retrodiffusion attenuee à partir de zref 
    '''
    Tr = betamol[idxref:].copy() #np.zeros_like(BetaMol[idx_ref:])
    Tr2 = np.zeros_like(betamol[idxref:])
    print(len(Tr), len(Tr2))

    for i,j in zip(range(1, len(Tr2)),range(idxref+1, len(alt))):
        Tr[i] = Tr[i-1] + alphamol[i]*(alt[j]-alt[j-1])
        Tr2[i] = np.exp(-2*Tr[i])

    betamol_Z0 = betamol.copy() #np.ones_like(BetaMol)   
    betamol_Z0[idxref+1:] = betamol[idxref+1:]*Tr2[1:]
    print(betamol_Z0[idxref])  
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
parser.add_argument("--clearskylist", "-l", type=str, help="list of clear sky data day", required=True)
parser.add_argument("--ztop", "-top", type=int, help="bound top of calibration height", required=True)
parser.add_argument("--zbottom", "-bottom", type=int, help="bound top of calibration height", required=True)
parser.add_argument("--wavelength", "-w", type=int, help="wavelength", required=True)
opts = parser.parse_args()
print(opts)

###_____Ouvrir le list des jours sélectionnés
with open(opts.clearskylist, 'r') as f:
    all_data = [line.strip() for line in f.readlines()]
    
metadata_line = all_data[:4]
listdays = all_data[4:]
lidars = metadata_line[1].split(': ')[1]

for l in listdays:
    ###____Ouvrir et lire les données Ipral brutes: range(km), channel, rcs, background 
    ipralpath0 = list(Path('/bdd/SIRTA/pub/basesirta/1a/ipral/', l).glob('ipral_1a_Lz1R15mF30sPbck_v01_*.nc'))[0]
    # ipralpath = Path('/bdd/SIRTA/pub/basesirta/1a/ipral/2020/02/06/ipral_1a_Lz1R15mF30sPbck_v01_20200206_000000_1440.nc')
    ipral_remove_cloud_profiles(dt.datetime.strptime(l.replace("/",""), '%Y%m%d'), 7000, 
                                                          ipralpath0, Path('/homedata/nmpnguyen/IPRAL/RF/',ipralpath0.name))
    ipralpath = Path('/homedata/nmpnguyen/IPRAL/RF/',ipralpath0.name)
    ipralraw = xr.open_dataset(ipralpath)
    
    ###____Range corrected signal (photocounting)
    w = opts.wavelength
    if w == 532:
        signal = ipralraw['rcs_17']
        bckgrd = ipralraw['bckgrd_rcs_17']
    else: 
        signal = ipralraw['rcs_13']
        bckgrd = ipralraw['bckgrd_rcs_13']
        
    ipralrange = ipralraw['range'].values
    ipralalt = ipralrange + ipralrange[0]
    iprcs = (signal/(ipralrange**2) - bckgrd)*(ipralrange**2)

    ###____Preparation Pression, Temperature et alt 
    # ipralsimulpath = sorted(Path('/homedata/nmpnguyen/IPRAL/RF').glob('ipral_1a_Lz1R15mF30sPbck_v01_20200206*simul.pkl'))
    ipralsimulpath = sorted(Path('/homedata/nmpnguyen/IPRAL/RF/').glob(ipralpath.name.split('.')[0]+'_simul.pkl'))[0]
    ipralsimul = pd.read_pickle(ipralsimulpath)
    pression = ipralsimul['pression'].unstack(level=1)
    ta = ipralsimul['ta'].unstack(level=1)

    ###____Average des données 
    time_to_average = '1H'

    '''
    Appliquer aux données Opar : nuage 21.01.2019 et ciel clair 17.06.2019
    1. Calculer le profil Pr2_z0 = Pr2[z]/Pr2[z0] puis calculer la valeur de son intégrale etre zmin et zmax
    '''
    ### Determiner z0
    zbottom = opts.zbottom #5000
    ztop = opts.ztop #7000
    Z0, idx_ref = get_altitude_reference(zbottom, ztop, ipralrange)
    zintervalle = np.where((ipralrange >= zbottom) & (ipralrange <= ztop))[0]

    Pr2_Z0 = iprcs/iprcs.isel(range=idx_ref)
    Pr2_integ = np.zeros(len(iprcs['time']))
    for z in zintervalle[:-1]:
        Pr2_integ = Pr2_integ + Pr2_Z0.isel(range=z).values*(ipralrange[z+1]-ipralrange[z])

    Pr2_integM = np.mean(Pr2_Z0[zintervalle].values)

    '''
    2. Calculer le profil BetaMol[z]*Tr2(AlphaMol(z))[z0] 
    Et calculer son integrale entre zmin et zmax
    '''
    AlphaMol, BetaMol = get_backscatter_mol(pression.values, ta.values, w*1e-3)
    print(BetaMol)
    BetaMol_Z0_v2 = np.array([get_backscatter_mol_attn_v2(AlphaMol[i,:], BetaMol[i,:], ipralrange) for i in range(len(iprcs['time']))])
    BetaMol_Z0 = np.array([get_backscatter_mol_attn_v1(AlphaMol[i,:], BetaMol[i,:], ipralrange, idx_ref) for i in range(len(iprcs['time']))])
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
    '''
    5. Calculer le résidus residus et l'intégrale sur l'altitude
    '''
    residus = np.zeros(len(iprcs['time']))
    for t in range(len(ipralrange)):
        residus = residus + np.nansum([Pr2_norm[:,z],- BetaMol_Z0[:,z]], axis=0)*(ipralrange[t]-ipralrange[t-1])

    fichier = open(Path('/homedata/nmpnguyen/IPRAL/RF/', l.replace("/",""), str(w), ipralpath.name.split('.')[0]+'_residus.txt'), 'a')
    fichier.write(f'calibration altitude: {[zbottom, ztop]}')
    fichier.write(f'wavelength: {w}')
    for i in range(len(residus)):
        fichier.write(f'\n{round(residus[i],6)},{str(iprcs.time[i].values)}')

    fichier.close()
    '''
    6. Plot un profil normalisé et moléculaire
    '''
    for n in range(1, len(iprcs['time']), 20):
        plt.clf()
        fig, (ax3, ax4) = plt.subplots(nrows=1, ncols=2, figsize=[12,6], sharey=True)#

        ax3.semilogx(Pr2_norm[n,:len(ipralrange)], ipralrange, label='signal normalisé')
        ax3.semilogx(BetaMol_Z0[n,:], ipralrange, label='signal moléculaire attn depuis z0')
        ax3.semilogx(BetaMol_Z0_v2[n,:], ipralrange, label='signal moleculaire attn depuis sol')
        ax3.axhspan(zbottom, ztop, color='y', alpha=0.5, lw=0, label='calibration height')
        leg3 = ax3.legend()
        leg3.set_title(f'4/ cRF = {cRF[n]}')
        ax3.set(xlabel='backscatter, 1/m.sr')
        ax3.set_ylim(0, 20000)

        ax4.plot(Pr2_norm[n,:len(ipralrange)]/BetaMol_Z0[n,:], ipralrange, label='signal SR, depuis Z0')
        ax4.plot(Pr2_norm[n,:len(ipralrange)]/BetaMol_Z0_v2[n,:], ipralrange, label='signal SR, depuis sol')
        ax4.axhspan(zbottom, ztop, color='y', alpha=0.5, lw=0, label='calibration height')
        ax4.vlines(1, ymin=ipralrange.min(), ymax=ipralrange.max(), color='k', zorder=10)
        # ax.set(title='5.916e-11')
        leg4 = ax4.legend()
        leg4.set_title(f'4/ cRF = {cRF[n]}')
        ax4.set_ylim(0, 20000)
        ax4.set(xlabel = 'scattering ratio')
        ax4.set_xlim(-1, 5)
        plt.suptitle(f'zmin = {zbottom}, zmax = {ztop}, z0 = {ipralrange[idx_ref]}\n IPRAL: {str(iprcs.time[n].values)}')
        plt.savefig(Path('/homedata/nmpnguyen/IPRAL/RF/', l.replace("/",""), str(w),  str(iprcs.time[n].values)+'RF_v1.png'))

### Préparer un profil de données gluing analog + photocounting depuis les produits SCC (utilisé le produit Hirellp
# scc_output_path = Path('/homedata/pietras/IPRAL/SCC/Input_auto/Output/Auto/375/')
# scc_output_list = sorted(scc_output_path.glob('**/elic/*20200708sir8018_*_v5.2.1.nc'))
# hirelpp_path = Path('/homedata/pietras/IPRAL/SCC/Input_auto/Output/Auto/375/20200708sir8018/hirelpp/sir_009_0000985_202007081800_202007081900_20200708sir8018_hirelpp_v5.2.1.nc')