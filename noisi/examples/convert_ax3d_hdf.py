import numpy as np
from netCDF4 import Dataset
import h5py
from glob import glob
from math import floor
from mpi4py import MPI
from scipy.signal import sosfilt, tukey
from obspy.signal.interpolation import lanczos_interpolation
from noisi.util.filter import lowpass

# - input --------------------------------------------------------------------
sourcegrid_file = "example/sourcegrid.npy"
f_out_name = "G.SSB..MXN.h5"
input_file_path = "/home/lermert/code/axisem3d_build/output/stations/axisem3d_synthetics.nc.rank*"
process_filter = {'type': 'lowpass',
                    'corners': 9,
                    'zerophase': False,
                    'freq_max': 0.1}
process_pad_factor = 10.
process_taper = {'type': 'tukey',
                 'alpha': 0.1,
                 'only_trace_end': True}
process_decimate = {'new_sampling_rate': 5.0,
                    'lanczos_window_width': 25.0}
channel = 'Z'
dquantity = 'DIS'
# - end input ----------------------------------------------------------------

channels = {'Z': 2, 'E': 0, 'N': 1}


def process_trace(trace, taper, sos, pad_factor, old_t0, old_sampling_rate,
                  new_sampling_rate, new_npts):
    # taper
    trace *= taper

    # pad
    n = len(trace)
    n_pad = int(pad_factor * n)
    padded_trace = np.zeros(n_pad)
    padded_trace[n_pad // 2: n_pad // 2 + n] = trace

    # filter
    if process_filter['zerophase']:
        firstpass = sosfilt(sos, padded_trace)
        padded_trace = sosfilt(sos, firstpass[:: -1])[:: -1]
    else:
        padded_trace = sosfilt(sos, padded_trace)
    # undo padding
    trace = padded_trace[n_pad // 2: n_pad // 2 + n]

    trace = lanczos_interpolation(trace, old_start=old_t0,
                                  old_dt=1. / old_sampling_rate,
                                  new_start=old_t0,
                                  new_dt=1. / new_sampling_rate,
                                  new_npts=new_npts,
                                  a=process_decimate['lanczos_window_width'],
                                  window='lanczos')
    return(trace)


# in case of several input files
input_files = glob(input_file_path)
input_file_0 = input_files[0]

# open input file
nc_syn = Dataset(input_file_0, 'r', format='NETCDF4')
# time step and n_time, Fs
var_time = nc_syn.variables['time_points']
nstep = len(var_time)
Fs = 1. / (np.mean(var_time[1:] - var_time[:-1]))
print("old sampling rate: ", Fs)
solver_dtype = var_time.datatype

# open source file
f_sources = np.load(sourcegrid_file)
# add some metainformation needed
comp = channel[-1]
ntraces = f_sources.shape[-1]
new_npts = floor((nstep / Fs) *
                 process_decimate['new_sampling_rate'])

# define filter and taper
taper = tukey(nstep, 0.1)
if process_taper['only_trace_end']:
    taper[: nstep // 2] = 1.

# filter
if process_filter['type'] == 'lowpass':
    sos = lowpass(freq=process_filter['freq_max'],
                  df=Fs,
                  corners=process_filter['corners'])
    # Important ! ! !
    # If using a Chebyshev filter as antialias, that is a good idea, but there
    # is a risk of accidentally cutting away desired parts of the spectrum
    # because rather than the corner frequency, the -96dB frequency is chosen
else:
    raise NotImplementedError('Filter {} is not implemented yet.'.format(
                              process_filter['type']))

# initialize output file

f_out = h5py.File(f_out_name, "w")

# DATASET NR 1: STATS
stats = f_out.create_dataset('stats', data=(0, ))
stats.attrs['data_quantity'] = dquantity
stats.attrs['ntraces'] = ntraces
stats.attrs['Fs'] = process_decimate['new_sampling_rate']
stats.attrs["fdomain"] = False
stats.attrs['nt'] = int(new_npts)

# DATASET NR 2: Source grid
sources = f_out.create_dataset('sourcegrid', data=f_sources[0:2])

# DATASET Nr 3: Seismograms itself
traces_h5 = f_out.create_dataset('data', (ntraces, new_npts),
                                 dtype=np.float32)


# read in traces
for i in range(ntraces):
    if i % 1000 == 0:
        print(i)
    try:
        tr = nc_syn.variables['SRC.%08g.ENZ' % i][:, channels[comp]]

    # process traces
        tr = process_trace(tr, taper, sos, process_pad_factor,
                           old_t0=var_time[0],
                           old_sampling_rate=Fs,
                           new_sampling_rate=process_decimate[
                           'new_sampling_rate'],
                           new_npts=new_npts)
    except KeyError:
        continue
    # save in new output file
    traces_h5[i, :] = tr

f_out.flush()

# if non-assembled input files: repeat for all the other input files:
if len(input_files) > 1:
    for ix_inp in range(len(input_files)[1:]):
        # open input file
        nc_syn = Dataset(input_files[ix_inp], 'r', format='NETCDF4')
        for i in range(ntraces):
            if i % 1000 == 0:
                print(i)
            try:
                tr = nc_syn.variables['SRC.%08g.ENZ' % i][:, channels[comp]]

                # process traces
                tr = process_trace(tr, taper, sos, process_pad_factor,
                               old_t0=var_time[0],
                               old_sampling_rate=Fs,
                               new_sampling_rate=process_decimate[
                               'new_sampling_rate'],
                               new_npts=new_npts)
            except KeyError:
                continue
            # save in new output file
            traces_h5[i, :] = tr
