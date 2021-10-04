#/usr/bin/env python
#encoding:utf-8

# Vincent Noel CNRS - 2019-10-22

from datetime import datetime
import numpy as np
import xarray as xr

def convert_to_datetime(date, h1):

    day, month, year = [int(x) for x in date.split('/')]
    hh, mm, ss = [int(x) for x in h1.split(':')]
    time = datetime(year, month, day, hh, mm, ss)
    return time


def read_teslas(f, debug=False):
    
    with open(f, 'rb') as fid:

        nb_nombre = 8
        nb_col = 7

        l1 = fid.readline()[:-2]
        l2 = fid.readline()[:-2]
        l3 = fid.readline()[:-2]

        l2s = str(l2).split()[1:-1]
        lieu, date1, h1, date2, h2, altsta, longitude, latitude, angle_zenith = l2s
        print(date1, h1, date2, h2)

        l3s = str(l3).split()[1:-1]
        
        if debug:
            print(f'line 3 : {str(l3)}')
        
        nb_laser = int(np.floor(len(l3s)-1)/2)
        nb_tir = np.zeros([nb_laser])
        freq = np.zeros([nb_laser])
        for i in np.r_[0:nb_laser]:
            nb_tir[i] = int(l3s[i*2])
            freq[i] = int(l3s[i*2+1])
        # apparently the number of channels is always in 5th position on header line 3
        # even if there are more than 2 channels
        nb_voies = int(l3s[4])

        if nb_voies < 1:
            print(f'nb_voies = {nb_voies}')
            print(l2)
            print(l3)
            raise('The header indicates a number of channel < 1')

        if debug:
            print(nb_laser, nb_tir, freq, nb_voies)

        par_read = np.zeros([nb_voies, nb_nombre])
        channel_names = []
        for i in np.r_[0:nb_voies]:
            ligne = fid.readline()[:-2]
            
            if debug:
                print(f'param line {i} : {ligne}')
            
            var = [x for x in str(ligne).split()[1:]]
            par_read[i,:7] = var[:7]
            channel_names.append(var[7])

        if np.any(par_read[:,3] != par_read[0,3]):
            raise('number of data different')

        if debug:
            print(f'Channel_names : {channel_names}')
            print(f'par_read = {par_read}')
            
        if ''.join(channel_names[2:6]).count('00355.o') == 4:
            channel_names[2] = '00355.o.Mid'
            channel_names[3] = '00355.o.High'
            channel_names[4] = '00355.o.Verylow'
            channel_names[5] = '00355.o.Low'
            
        npts = int(par_read[0,3])
        sig_matrix = np.zeros([1, npts, nb_voies])
        
        for i in np.r_[0:nb_voies]:
            if debug:
                print(f'Reading channel {i}')

            ligne = fid.readline()
            if debug:
                print(f'ligne bidon = {ligne}')

            data = np.fromfile(fid, dtype=np.int32, count=npts)
            sig_matrix[0,:,i] = data
    
    zrange = np.arange(npts) * par_read[0,6] / 1e3
    xrange = xr.DataArray(zrange, dims='range')
    xrange.attrs['units'] = 'km'

    xchannels = xr.DataArray(channel_names, dims='channel')

    time = [convert_to_datetime(date1, h1)]
    if debug:
        print(date1)
    time = np.array(time)
    time = time.astype('datetime64[ms]')
    
    xdata = xr.DataArray(sig_matrix, coords={'time': time, 'range':xrange, 'channel':xchannels}, dims=['time', 'range', 'channel'])
    xdata['time'].encoding['units'] = 'seconds since 2000-01-01'
    xdata.attrs['alt_station(km)'] = float(altsta) / 1e3
    xdata.attrs['start_time'] = str(convert_to_datetime(date1, h1))
    xdata.attrs['end_time'] = str(convert_to_datetime(date2, h2))
    xdata.name = 'signal'
    xdata.attrs['long_name'] = 'signal'
    xdata.attrs['location'] = lieu
    xdata.attrs['longitude'] = longitude
    xdata.attrs['latitude'] = latitude
    xdata.attrs['angle_zenith'] = angle_zenith

    return xdata


def main(f):
	read_teslas(f, debug=True)


if __name__=='__main__':
    import plac
    plac.call(main)