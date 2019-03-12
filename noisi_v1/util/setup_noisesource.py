# coding: utf-8
import numpy as np
from obspy.geodetics import gps2dist_azimuth
from obspy.signal.invsim import cosine_taper
import matplotlib.pyplot as plt
import h5py
from noisi_v1 import WaveField
import json
from glob import glob
import os
from scipy.fftpack import next_fast_len
from scipy.signal import iirfilter
from scipy.signal import freqz_zpk
from noisi_v1.util.geo import get_spherical_surface_elements

##################################################################
# USER INPUT
##################################################################
# path to project and source model
projectpath = '../'
sourcepath = '.'

# geography - a sequence of distributions 'homogeneous', 'ocean',
# 'gaussian_blob' in any order. The order has to match with the 
# order od the list of spectra in params_spectra, i.e. the first 
# distribution will be assigned the first spectrum, the second 
# distribution the second spectrum, etc. 
# Similarly, the first 'gaussian_blob' will be assigned the first
# set of parameters in params_gaussian_blobs, and so on.
distributions = ['homogeneous']

# Resolution of the coastlines (only relevant for ocean distributions)
# (see basemap documentation)
# Use coarser for global and finer for regional models
coastres = 'c'

# Geographic gaussian blobs. Will only be used if 'gaussian_blob'
# is found in the list of distributions. Will be used
# in order of appearance
params_gaussian_blobs = [{'center': (-10., 0.), 'sigma_radius_m': 2000000.,
                          'rel_weight': 2., 'only_ocean': True}]


# Further parameters are pulled out of the measr_config file.
###############################################################################

grd = np.load(os.path.join(projectpath, 'sourcegrid.npy'))
ntraces = np.shape(grd)[-1]
print('Loaded source grid')


config = json.load(open(os.path.join(projectpath, 'config.json')))
source_config = json.load(open(os.path.join(sourcepath, 'source_config.json')))
measr_config = json.load(open(os.path.join(sourcepath, 'measr_config.json')))
print('Loaded config files.')


if source_config['preprocess_do']:
    ext = '*.h5'
    wavefield_path = os.path.join(sourcepath, 'wavefield_processed')
else:
    ext = '*.h5'
    wavefield_path = config['wavefield_path']


wfs = glob(os.path.join(wavefield_path, ext))
if wfs != []:
    print('Found wavefield.')
    with WaveField(wfs[0]) as wf:
        df = wf.stats['Fs']
        nt = wf.stats['nt']

else:
    df = float(input('Sampling rate of synthetic Greens functions in Hz?\n'))
    nt = int(input('Nr of time steps in synthetic Greens functions?\n'))





#s for the fft is larger due to zeropadding --> apparent higher frequency sampling\n",
    # n = next_fast_len(2*nt-1)
n = next_fast_len(2 * nt - 1)
freq = np.fft.rfftfreq(n, d=1. / df)
taper = cosine_taper(len(freq), 0.01)
print('Determined frequency axis.')


def get_distance(grid, location):
    def f(lat, lon, location):
        return abs(gps2dist_azimuth(lat, lon, location[0], location[1])[0])

    dist = np.array([f(lat, lon, location) for lat, lon in zip(grid[1],
                                                               grid[0])])
    return dist


def get_ocean_mask():
    # Use Basemap to figure out where ocean is
    print('Getting ocean mask...')
    from mpl_toolkits.basemap import Basemap
    latmin = grd[1].min()
    latmax = grd[1].max()
    lonmin = grd[0].min()
    lonmax = grd[0].max()
    print("Latitude {}--{},\n\
Longitude {}--{}".format(round(latmin, 2),
                         round(latmax, 2),
                         round(lonmin, 2), round(lonmax, 2)))
    m = Basemap(rsphere=6378137, resolution=coastres, projection='cea',
                llcrnrlat=latmin, urcrnrlat=latmax,
                llcrnrlon=lonmin, urcrnrlon=lonmax)
    (east, north) = m(grd[0], grd[1])

    ocean_mask = [not m.is_land(x, y) for (x, y) in zip(east, north)]
    return np.array(ocean_mask)


def get_geodist(disttype, gaussian_params=None):

    if disttype == 'gaussian':
        dist = get_distance(grd, gaussian_params['center'])
        gdist = np.exp(-(dist) ** 2 /
                       (2 * gaussian_params['sigma_radius_m'] ** 2))

        if gaussian_params['only_ocean']:
            if 'ocean_mask' not in locals():
                ocean_mask = get_ocean_mask()
            gdist *= ocean_mask

        return gdist

    elif disttype == 'homogeneous':
        return np.ones(ntraces)

    elif disttype == 'ocean':
        if 'ocean_mask' not in locals():
            ocean_mask = get_ocean_mask()
        return ocean_mask


def get_spectrum(sparams):
    spec = taper * np.exp(-(freq - sparams['central_freq']) ** 2 /
                          (2 * sparams['sigma_freq'] ** 2))
    return spec / np.max(np.abs(spec))


def get_specbasis(bandpass):

    low = bandpass[0]
    high = bandpass[1]
    corners = bandpass[2]

    low = low / (0.5 * df)
    high = high / (0.5 * df)

    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    w, h = freqz_zpk(z, p, k, worN=len(freq))

    # always zerophase
    h2 = h * np.conjugate(h)
    return np.real(h2)

#########################
# Create the source distr
#########################


#########################
# geography
#########################
num_bases = len(distributions)
gauss_cnt = 0
basis_geo = np.zeros((num_bases,ntraces))

print('Filling distribution...')

for i in range(num_bases):

    if distributions[i] =='gaussian':

        gaussparams = params_gaussian_blobs[gauss_cnt]
        gauss_cnt += 1
        basis_geo[i,:] = get_geodist('gaussian',gaussparams)

    elif distributions[i] in ['ocean','homogeneous']:

        basis_geo[i,:] = get_geodist(distributions[i])
        
    
    else:
        print(distributions)
        raise NotImplementedError('Unknown geographical distributions. \
            Must be \'gaussian\', \'homogeneous\' or \'ocean\'.')

try:
    print('Plotting...')
    from noisi_v1.util import plot
    for i in range(num_bases):
        plot.plot_grid(grd[0],grd[1],basis_geo[i,:],normalize=False,
        outfile = os.path.join(sourcepath,'geog_distr_basis{}.png'.format(i)))
except ImportError:
    print('Plotting not possible (is basemap installed?)')

#########################
# spectrum
#########################

if measr_config['bandpass'] is None:
    basis_spec = np.array(np.ones(len(freq)), dmin=2)

else:
    num_sbases = len(measr_config['bandpass'])
    basis_spec = np.zeros((num_sbases, len(freq)))
    for i in range(num_sbases):
        basis_spec[i, :] = get_specbasis(measr_config['bandpass'][i])


plt.figure()
for i in range(basis_spec.shape[0]):
    plt.semilogx(freq, basis_spec[i, :], '--')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Source power (scaled)')
plt.savefig(os.path.join(sourcepath, 'freq_distr_startingmodel.png'))


########################
# Initial geographic
# weighting (unif)
########################

weights = np.eye(basis_spec.shape[0], num_bases)

########################
# Save to an hdf5 file
########################
with h5py.File(os.path.join(sourcepath, 'step_0',
                            'starting_model.h5'), 'w') as fh:
    fh.create_dataset('coordinates', data=grd.astype(np.float64))
    fh.create_dataset('frequencies', data=freq.astype(np.float64))
    fh.create_dataset('distr_basis', data=basis_geo.astype(np.float64))

    # for now: Geographic model can vary freely.
    fh.create_dataset('distr_weights', data=weights)
    fh.create_dataset('spect_basis', data=basis_spec.astype(np.float64))
#    fh.create_dataset('surf_areas', data=surf_areas.astype(np.float64))

basis1_b = np.ones(basis_geo.shape)
with h5py.File(os.path.join(sourcepath, 'step_0', 'base_model.h5'), 'w') as fh:
    fh.create_dataset('coordinates', data=grd.astype(np.float32))
    fh.create_dataset('frequencies', data=freq.astype(np.float32))
    fh.create_dataset('distr_basis', data=basis1_b.astype(np.float32))
    fh.create_dataset('distr_weights', data=weights.astype(np.float32))
    fh.create_dataset('spect_basis', data=basis_spec.astype(np.float32))
    #fh.create_dataset('surf_areas',data=surf_areas.astype(np.float64))

print('Done.')
