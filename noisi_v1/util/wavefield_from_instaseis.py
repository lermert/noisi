# create a wavefield from instaseis
from mpi4py import MPI
import instaseis
import h5py
import os
from pandas import read_csv
import numpy as np
import json
from noisi_v1.util.geo import geograph_to_geocent
from obspy.geodetics import gps2dist_azimuth

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# get config
source_config = json.load(open('source_config.json'))
config = json.load(open('../config.json'))
Fs = source_config['sampling_rate']
path_to_db = config['wavefield_path']
channel = source_config['channel']

# read sourcegrid
f_sources = np.load(os.path.join(os.path.dirname(os.getcwd()),
                                 'sourcegrid.npy'))
ntraces = f_sources.shape[-1]

# open the database
db = instaseis.open_db(path_to_db)

# get: synthetics duration and sampling rate in Hz
stest = db.get_seismograms(source=instaseis.ForceSource(latitude=0.0,
                           longitude=0.0),
                           receiver=instaseis.Receiver(latitude=10.,
                           longitude=0.0),
                           dt=1. / source_config['sampling_rate'])[0]
ntimesteps = stest.stats.npts


# read station from file
stationlist = read_csv(os.path.join(os.path.dirname(os.getcwd()),
                                    'stationlist.csv'))
net = stationlist.at[rank, 'net']
sta = stationlist.at[rank, 'sta']
lat = stationlist.at[rank, 'lat']
lon = stationlist.at[rank, 'lon']
print(net, sta, lat, lon)

# output directory:
if rank == 0:
    os.system('mkdir -p wavefield_processed')

f_out_name = '{}.{}..{}.h5'.format(net, sta, channel)
f_out_name = os.path.join('wavefield_processed', f_out_name)


if not os.path.exists(f_out_name):

    startindex = 0

    f_out = h5py.File(f_out_name, "w")
    # DATASET NR 1: STATS
    stats = f_out.create_dataset('stats', data=(0, ))
    stats.attrs['reference_station'] = '{}.{}'.format(net, sta)
    stats.attrs['data_quantity'] = config['synt_data']
    stats.attrs['ntraces'] = ntraces
    stats.attrs['Fs'] = Fs
    stats.attrs['nt'] = int(ntimesteps)

    # DATASET NR 2: Source grid
    sources = f_out.create_dataset('sourcegrid', data=f_sources[0: 2])
    lat1 = geograph_to_geocent(float(lat))
    lon1 = float(lon)
    rec1 = instaseis.Receiver(latitude=lat1, longitude=lon1)

    # DATASET Nr 3: Seismograms itself
    traces = f_out.create_dataset('data', (ntraces, ntimesteps),
                                  dtype=np.float32)
    if channel[-1] == 'Z':
        c_index = 0
    elif channel[-1] == 'R':
        c_index = 1
    elif channel[-1] == 'T':
        c_index = 2

else:
    f_out = h5py.File(f_out_name, "r+")
    startindex = len(f_out['data'])


# jump to the beginning of the trace in the binary file
for i in range(startindex, ntraces):
    if i % 1000 == 1:
        print('Converted %g of %g traces' % (i, ntraces))

    lat_src = geograph_to_geocent(f_sources[1, i])
    lon_src = f_sources[0, i]

    if c_index in [1, 2]:
        fsrc = instaseis.ForceSource(latitude=lat_src,
                                     longitude=lon_src,
                                     f_t=1.e09, f_p=1.e09)
    elif c_index == 0:
        fsrc = instaseis.ForceSource(latitude=lat_src,
                                     longitude=lon_src, f_r=1.e09)

    if config['synt_data'] == 'DIS':
        values = db.get_seismograms(source=fsrc, receiver=rec1, dt=1. / Fs)
    elif config['synt_data'] == 'VEL':
        values = db.get_seismograms(source=fsrc, receiver=rec1, dt=1. / Fs,
                                    kind='velocity')
    elif config['synt_data'] == 'ACC':
        values = db.get_seismograms(source=fsrc, receiver=rec1, dt=1. / Fs,
                                    kind='acceleration')
    else:
        raise ValueError('Unknown data quantity {}. Choose DIS, VEL or ACC in\
 config.json.'.format(config['synt_data']))

    if c_index in [1, 2]:
        baz = gps2dist_azimuth(lat_src, lon_src, lat, lon)[2]
        values.rotate('NE->RT', back_azimuth=baz)

    values = values[c_index]

    # Save in traces array
    traces[i, :] = values.data

f_out.close()
