# -*- coding: utf-8 -*-

'''
wavefield for noisi from axisem3d
'''
from mpi4py import MPI
import h5py
import numpy as np
from netCDF4 import Dataset
from surface_utils import *
import json
import os
from noisi.util.geo import geograph_to_geocent
from noisi.util.filter import *
from obspy.signal.interpolation import lanczos_interpolation
try:
    from scipy.signal import sosfilt
except ImportError:
    from obspy.signal._sosfilt import _sosfilt as sosfilt
try:
    from scipy.signal import tukey
except ImportError:
    print("Sorry, no Tukey taper (old scipy version).")
    from obspy.signal.invsim import cosine_taper
from math import ceil, floor
import sys

# ----------------------------------------------------------------------------
# input
# ----------------------------------------------------------------------------

input_file = "axisem3d_surface.nc"
sourcegrid_file = "sourcegrid.npy"
f_out_name = "G.SSB..MXZ.h5"
# n_chunk: if the seismogram is long with many spatial Fourier coeffs,
# break it down into n_chunk > 1 pieces so that it fits into memory
n_chunk = 1
process_filter = {'type': 'bandpass',
                    'corners': 4,
                    'zerophase': True,
                    'freq_max': 1.0,
		             'freq_min': 0.01}
# zero pad the seismograms to avoid edge effects from filtering
process_pad_factor = 4.
# taper applied to trace end because the seismogram 
# "stops" wherever the simulation stops
process_taper = {'type': 'tukey',
                 'alpha': 0.1,
                 'only_trace_end': True}
process_decimate = {'new_sampling_rate': 5.0,
                    'lanczos_window_width': 20.0}
channel = 'MXZ'
dquantity = 'DIS' #DIS (-placement), VEL (-ocity) or ACC (-celeration).

# ----------------------------------------------------------------------------
# end of input
# ----------------------------------------------------------------------------



comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def get_trace_from_netcdf(nc_surf, lon, lat, slvdtype, comp='Z'):
    # global attribute
    srclat = nc_surf.source_latitude
    srclon = nc_surf.source_longitude
    # srcdep = nc_surf.source_depth
    srcflat = nc_surf.source_flattening
    surfflat = nc_surf.surface_flattening
    # time
    var_time = nc_surf.variables['time_points']
    nstep = len(var_time)
    solver_dtype = var_time.datatype
    # theta
    var_theta = nc_surf.variables['theta']
    nele = len(var_theta)
    # GLL and GLJ
    var_GLL = nc_surf.variables['GLL']
    var_GLJ = nc_surf.variables['GLJ']
    nPntEdge = len(var_GLL)

    # distance from source to receiver
    srctheta, srcphi = latlon2thetaphi(srclat, srclon, srcflat)

    rmat = rotation_matrix(srctheta, srcphi)

    theta, phi = latlon2thetaphi(lat, lon, surfflat)
    xglb = thetaphi2xyz(theta, phi)
    xsrc = rmat.T.dot(xglb)

    dist, azimuth = xyz2thetaphi(xsrc)
    d, az, baz = gps2dist_azimuth(srclat, srclon, lat, lon, a=1., f=surfflat)
    baz = np.radians(baz)

    sindist = np.sin(dist)
    cosdist = np.cos(dist)
    sinbaz = np.sin(baz)
    cosbaz = np.cos(baz)

    # locate receiver
    eleTag = -1
    for iele in np.arange(0, nele):
        if dist <= var_theta[iele, 1]:
            eleTag = iele
            break
    assert eleTag >= 0, 'Fail to locate receiver at %f, %f deg, dist = %f' \
        % (lat, lon, dist)

    theta0 = var_theta[eleTag, 0]
    theta1 = var_theta[eleTag, 1]
    eta = (dist - theta0) / (theta1 - theta0) * 2. - 1.

    # weights considering axial condition
    if eleTag == 0 or eleTag == nele - 1:
        weights = interpLagrange(eta, var_GLJ)
    else:
        weights = interpLagrange(eta, var_GLL)
    # Fourier
    # NOTE: change to stepwise if memory issue occurs
    fourier_r = nc_surf.variables['edge_' + str(eleTag) + 'r']
    fourier_i = nc_surf.variables['edge_' + str(eleTag) + 'i']
    fourier = fourier_r[:, :] + fourier_i[:, :] * 1j

    nu_p_1 = int(fourier_r[:, :].shape[1] / nPntEdge / 3)
    exparray = 2. * np.exp(np.arange(0, nu_p_1) * 1j * azimuth)
    exparray[0] = 1.

    # compute disp
    disp = np.zeros((nstep, 3), dtype=solver_dtype)

    size_chunk = nstep // n_chunk + 1
    start_chunks = [size_chunk * i for i in range(n_chunk)]
    for s_chnk in start_chunks:
        # where does the chunk end and how long is it?
        e_chnk = min(s_chnk + size_chunk, nstep)
        n_ths_chnk = e_chnk - s_chnk
        spz = np.zeros((n_ths_chnk, 3))
        #for istep in np.arange(0, nstep):
    #    spz = np.zeros(3)
        for idim in np.arange(0, 3):
            start = idim * nPntEdge * nu_p_1
            end = idim * nPntEdge * nu_p_1 + nPntEdge * nu_p_1
            # fmat = fourier[istep, start:end].reshape(nPntEdge, nu_p_1)
            # spz[idim] = weights.dot(fmat.dot(exparray).real)
            fmat = fourier[s_chnk: e_chnk, start: end].reshape(n_ths_chnk,
                nPntEdge, nu_p_1)
            # this reshape takes sequential sub-arrays of length nu_p_1 from each row and puts them into new rows one after another
            #print(fmat.shape)
            temp = fmat.dot(exparray).real
            #temp.reshape(nPntEdge, n_ths_chnk)
            #print(temp.shape)
            #spz[:, idim] = weights.dot(temp)
            spz[:, idim] = weights.dot(temp.T)

            # radial, transverse displacelement
            # ur = spz[0] * np.sin(dist) + spz[2] * np.cos(dist)
            # ut = spz[0] * np.cos(dist) - spz[2] * np.sin(dist)
            if comp == 'E':
                ut = spz[:, 0] * cosdist - spz[:, 2] * sindist
                disp[s_chnk:e_chnk, 0] = -ut * sinbaz + spz[1] * cosbaz

            elif comp == 'N':
                ut = spz[:, 0] * cosdist - spz[:, 2] * sindist
                disp[s_chnk:e_chnk, 1] = -ut * cosbaz - spz[1] * sinbaz

            elif comp == 'Z':
                ur = spz[:, 0] * sindist + spz[:, 2] * cosdist
                disp[s_chnk:e_chnk, 2] = ur
            else:
                raise ValueError('Components: E, N or Z')
    return(np.array(disp))

def process_trace(trace, taper, sos, pad_factor, old_t0, old_sampling_rate, new_sampling_rate, new_npts):

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
    trace = np.asarray(trace, order='C')
    trace = lanczos_interpolation(trace, old_start=old_t0, 
                                  old_dt=1./old_sampling_rate,
                                  new_start=old_t0, 
                                  new_dt=1./new_sampling_rate,
                                  new_npts=new_npts,
                                  a=process_decimate['lanczos_window_width'],
                                  window='lanczos')
    return(trace)


# open surface database
nc_surf = Dataset(input_file, 'r', format='NETCDF4')
# time step and n_time
var_time = nc_surf.variables['time_points']
nstep = len(var_time)
Fs = 1. / (np.mean(var_time[1:] - var_time[:-1]))
print("old sampling rate: ", Fs)
solver_dtype = var_time.datatype

# get config
comp = channel[-1]
# read sourcegrid
f_sources = np.load(sourcegrid_file)
ntraces = f_sources.shape[-1]
new_npts = floor((nstep / Fs) *
                   process_decimate['new_sampling_rate'])

f_out_name = os.path.join('wavefield_processed', f_out_name)
if os.path.exists(f_out_name):
    raise ValueError("File %s exists already." % f_out_name)

comm.barrier()
if rank == 0:
    os.system('mkdir -p wavefield_processed')
    f_out = h5py.File(f_out_name, "w")

    # DATASET NR 1: STATS
    stats = f_out.create_dataset('stats', data=(0, ))
    stats.attrs['data_quantity'] = dquantity
    stats.attrs['ntraces'] = ntraces
    stats.attrs['Fs'] = process_decimate['new_sampling_rate']

    stats.attrs['nt'] = int(new_npts)

    # DATASET NR 2: Source grid
    sources = f_out.create_dataset('sourcegrid', data=f_sources[0:2])

    # DATASET Nr 3: Seismograms itself
    traces_h5 = f_out.create_dataset('data', (ntraces, new_npts),
                                        dtype=np.float32)

# Define processing
# half-sided tukey taper
try:
    taper = tukey(nstep, 0.1)
except NameError:
    taper = cosine_taper(nstep, 0.1)

if process_taper['only_trace_end']:
    taper[: nstep // 2] = 1.

# filter
if process_filter['type'] == 'lowpass':
    sos = lowpass(freq=process_filter['freq_max'],
                  df=Fs,
                  corners=process_filter['corners'])
elif process_filter['type'] == 'cheby2_lowpass':
    sos = cheby2_lowpass(freq=process_filter['freq_max'],
                  df=Fs,
                  maxorder=process_filter['max_order'])
elif process_filter['type'] == 'bandpass':
    sos = bandpass(freqmin=process_filter['freq_min'],
                   freqmax=process_filter['freq_max'],
		   corners=process_filter['corners'],df=Fs)
else:
    raise NotImplementedError('Filter {} is not implemented yet.'.format(
                    process_filter['type']))

old_t0 = var_time[0]
old_sampling_rate = Fs
new_sampling_rate = process_decimate['new_sampling_rate']

print("Hello, this is rank %g." % rank)
traces = np.zeros((int(ceil(ntraces / size)), new_npts))
local_count = 0
for i in range(ntraces)[rank::size]:
    if i % 1000 == rank:
        print('%g / Converted %g of %g traces' % (rank, i, ntraces))
        sys.stdout.flush()
    # read station name, copy to output file

    lat_src = f_sources[1, i]#geograph_to_geocent(f_sources[1, i])
    lon_src = f_sources[0, i]

    values = get_trace_from_netcdf(nc_surf, lon_src, lat_src,
                                   solver_dtype, comp)[:,2]

    values = process_trace(values, taper, sos, process_pad_factor,
                           old_t0, old_sampling_rate, new_sampling_rate,
                           new_npts)

    if dquantity in ['VEL', 'ACC']:
        new_fs = process_decimate['new_sampling_rate']
        values = np.gradient(values, new_npts * [1. / new_sampling_rate])
        if dquantity == 'ACC':
            values = np.gradient(values, new_npts * [1. / new_sampling_rate])
    # Save in traces array
    traces[local_count, :] = values
    local_count += 1

# save in temporary npy file
np.save('traces_%g.npy' % rank, traces[:local_count])

comm.barrier()

# rank 0: combine
if rank == 0:
    #global_count = 0
    for i in range(size):
        traces = np.load('traces_%g.npy' % i)
        traces_h5[i::size,:] = traces
        #global_count += traces.shape[0]
    f_out.close()
