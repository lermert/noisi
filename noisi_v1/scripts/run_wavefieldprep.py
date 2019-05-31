import io
import yaml
import os
from pandas import read_csv
from warnings import warn
import numpy as np
from math import pi
from scipy.signal import hann
import h5py
from obspy.geodetics import gps2dist_azimuth
from noisi_v1 import WaveField
from noisi_v1.util.geo import geograph_to_geocent
from glob import glob
from mpi4py import MPI
import instaseis
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


class precomp_wavefield(object):
    """
    Class to handle the precomputation of wave field for noise correlation
    simulation. This can be done either through instaseis, or with analytic
    Green's functions. See noisi docs for details.
    """

    def __init__(self, args):

        self.args = args
        # add configuration
        configfile = os.path.join(args.project_path, 'config.yml')
        with io.open(configfile, 'r') as fh:
            config = yaml.safe_load(fh)
        self.config = config

        # create output folder
        self.wf_path = os.path.join(args.project_path, 'greens')
        if not os.path.exists(self.wf_path):
            os.mkdir(self.wf_path)

        # sourcegrid
        self.sourcegrid = np.load(os.path.join(self.args.project_path,
                                               'sourcegrid.npy'))

        # add stations
        self.stations = read_csv(os.path.join(args.project_path,
                                              'stationlist.csv'))
        self.prepare()

        # run: instaseis or analytic
        if self.config['wavefield_type'] == 'analytic':
            self.function = self.green_analytic
        elif self.config['wavefield_type'] == 'instaseis':
            self.function = self.green_from_instaseis
        elif self.config['wavefield_type'] == 'custom':
            warn('Cannot prepare custom wave field,\
 choose instaseis or analytic or prepare your own.')
        else:
            raise ValueError('Unknown wavefield_type ' +
                             config['wavefield_type'])

    def precompute(self):
        stations = list(self.stations.iterrows())
        for i, station in stations[rank: len(self.stations): size]:
            self.function(station)

        if rank == 0:
            wfile = glob(os.path.join(self.args.project_path,
                                      'greens', '*' +
                                      station['sta'] + '.*.h5'))[0]
            ofile = os.path.join(self.args.project_path,
                                 'wavefield_example.png')
            with WaveField(wfile) as wf:
                wf.plot_snapshot(self.npts / self.Fs * 0.3, outfile=ofile,
                                 stations=[(station['lat'],
                                            station['lon'])])

    def prepare(self):

        # initialize parameters
        self.Fs = self.config['wavefield_sampling_rate']
        self.npts = int(self.config['wavefield_duration'] / self.Fs)
        self.ntraces = self.sourcegrid.shape[-1]
        self.data_quantity = self.config['synt_data']
        self.freq = np.fft.rfftfreq(2 * self.npts, d=1.0 / self.Fs)

        # Apply a freq. domain taper to suppress high and low frequencies.
        freq_max = self.Fs / 2.0  # Nyquist
        if freq_max < self.config['wavefield_filter'][1]:
            warn("Selected upper filter corner is above Nyquist.\
 It will be reset.")
        freq_min = 1. / self.config['wavefield_duration']  # lowest resolved
        freq_max = min(freq_max, self.config['wavefield_filter'][1])
        freq_min = min(freq_min, self.config['wavefield_filter'][0])
        self.filt = [freq_min, freq_max]

        if self.config['wavefield_type'] == 'instaseis':
            path_to_db = self.config['wavefield_path']
            self.db = instaseis.open_db(path_to_db)
            if self.db.info['length'] < self.npts / self.Fs:
                warn("Resetting wavefield duration to axisem database length.")
                self.npts = self.db.info['length'] * self.Fs

    def green_from_instaseis(self, station):

        # set some parameters
        lat_sta = station['lat']
        lon_sta = station['lon']
        lat_sta = geograph_to_geocent(float(lat_sta))
        lon_sta = float(lon_sta)
        rec = instaseis.Receiver(latitude=lat_sta, longitude=lon_sta)
        point_f = float(self.config['wavefield_point_force'])

        station_id = station['net'] + '.' + station['sta'] + '..MX' + \
            self.config['wavefield_channel']
        if self.config['verbose']:
            print(station_id)

        if self.config['wavefield_channel'] == 'Z':
            c_index = 0
        elif self.config['wavefield_channel'] == 'R':
            raise NotImplementedError('Horizontal components not yet implem.')
        elif self.config['wavefield_channel'] == 'T':
            raise NotImplementedError('Horizontal components not yet implem.')

        # initialize the file
        f_out = os.path.join(self.wf_path, station_id + '.h5')

        with h5py.File(f_out, "w") as f:

            # DATASET NR 1: STATS
            stats = f.create_dataset('stats', data=(0,))
            stats.attrs['reference_station'] = station['sta']
            stats.attrs['data_quantity'] = self.data_quantity
            stats.attrs['ntraces'] = self.ntraces
            stats.attrs['Fs'] = self.Fs
            stats.attrs['nt'] = int(self.npts)

            # DATASET NR 2: Source grid
            f.create_dataset('sourcegrid', data=self.sourcegrid)

            # DATASET Nr 3: Seismograms itself
            traces = f.create_dataset('data', (self.ntraces, self.npts),
                                      dtype=np.float32)

            for i in range(self.ntraces):
                if i % 1000 == 0 and i > 0 and self.config['verbose']:
                    print('Converted %g of %g traces' % (i, self.ntraces))

                lat_src = geograph_to_geocent(self.sourcegrid[1, i])
                lon_src = self.sourcegrid[0, i]

                fsrc = instaseis.ForceSource(latitude=lat_src,
                                             longitude=lon_src, f_r=point_f)
                if self.config['synt_data'] == 'DIS':
                    values = self.db.get_seismograms(source=fsrc,
                                                     receiver=rec,
                                                     dt=1. / self.Fs)
                elif self.config['synt_data'] == 'VEL':
                    values = self.db.get_seismograms(source=fsrc,
                                                     receiver=rec,
                                                     dt=1. / self.Fs,
                                                     kind='velocity')
                elif self.config['synt_data'] == 'ACC':
                    values = self.db.get_seismograms(source=fsrc,
                                                     receiver=rec,
                                                     dt=1. / self.Fs,
                                                     kind='acceleration')
                else:
                    raise ValueError('Unknown data quantity. \
Choose DIS, VEL or ACC in configuration.')

                traces[i, :] = values[c_index][0: self.npts]

    def green_spec_analytic(self, distance_in_m):

        v_phase = self.args.v
        q = self.args.q
        rho = self.args.rho
        freq = self.freq
        w = 2 * pi * freq
        g_fd = np.zeros(freq.shape, dtype=np.complex)
        f = float(self.config['wavefield_point_force'])

        # evaluate the Greens fct.
        if self.data_quantity == 'DIS':
            fac1 = -1j * 1. / (4. * rho * v_phase ** 2)
        elif self.data_quantity == 'VEL':
            fac1 = w[1:] * 1. / (4. * rho * v_phase ** 2)

        fac2 = np.sqrt((2. * v_phase) / (pi * w[1:] * distance_in_m))
        phase = -1j * w[1:] / v_phase * distance_in_m + 1j * pi / 4.0
        decay = -(w[1:] * distance_in_m) / (2. * v_phase * q)

        g_fd[1:] = f * fac1 * fac2 * np.exp(phase) * np.exp(decay)

        return(g_fd)

    def green_analytic(self, station, verbose=False):

        # set some parameters
        lat_sta = station['lat']
        lon_sta = station['lon']
        station_id = station['net'] + '.' + station['sta'] + '..MX' + \
            self.config['wavefield_channel']
        if self.config['verbose']:
            print(station_id)

        # initialize the file
        f_out = os.path.join(self.wf_path, station_id + '.h5')

        with h5py.File(f_out, "w") as f:

            # DATASET NR 1: STATS
            stats = f.create_dataset('stats', data=(0,))
            stats.attrs['reference_station'] = station['sta']
            stats.attrs['data_quantity'] = self.data_quantity
            stats.attrs['ntraces'] = self.ntraces
            stats.attrs['Fs'] = self.Fs
            stats.attrs['nt'] = int(self.npts)

            # DATASET NR 2: Source grid
            f.create_dataset('sourcegrid', data=self.sourcegrid)

            # DATASET Nr 3: Seismograms itself
            f.create_dataset('data', (self.ntraces, self.npts),
                             dtype=np.float32)

            # loop over source locations
            for i in range(self.ntraces):
                lat = self.sourcegrid[1, i]
                lon = self.sourcegrid[0, i]
                r = gps2dist_azimuth(lat, lon, lat_sta, lon_sta)[0]

                g_fd = self.green_spec_analytic(r)

                # apply the freq. domain taper
                taper = np.zeros(self.freq.shape)
                i0 = np.argmin(np.abs(self.freq - self.filt[0]))
                i1 = np.argmin(np.abs(self.freq - self.filt[1]))
                taper[i0: i1] = hann(i1 - i0)

            # transform back to time domain
                g1_td_taper = np.fft.irfft(taper * g_fd)[0: self.npts]

            # write the result
                f['data'][i, :] = g1_td_taper
                f.flush()

            # close output file
            f.close()
        return()
