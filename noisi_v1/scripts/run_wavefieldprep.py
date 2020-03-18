import io
import yaml
import os
from pandas import read_csv
from warnings import warn
import numpy as np
from math import pi
import h5py
from obspy.geodetics import gps2dist_azimuth
from noisi_v1 import WaveField
from scipy.fftpack import next_fast_len
from scipy.signal import lfilter, butter
from noisi_v1.util.geo import geograph_to_geocent
from glob import glob
try:
    import instaseis
except ImportError:
    pass


class precomp_wavefield(object):
    """
    Class to handle the precomputation of wave field for noise correlation
    simulation. This can be done either through instaseis, or with analytic
    Green's functions. See noisi docs for details.
    """

    def __init__(self, args, comm, size, rank):

        self.args = args
        self.rank = rank
        self.size = size
        # add configuration
        configfile = os.path.join(args.project_path, 'config.yml')
        with io.open(configfile, 'r') as fh:
            config = yaml.safe_load(fh)
        self.config = config

        if self.config['wavefield_type'] == 'instaseis'\
           and not 'instaseis' in globals():
            raise ImportError('Module instaseis was not found.')

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
            raise ValueError('Cannot prepare custom wave field,\
 choose \'instaseis\' or \'analytic\'.')
        else:
            raise ValueError('Unknown wavefield_type ' +
                             config['wavefield_type'] +
                             " select \"instaseis\" or \"analytic\"")

    def precompute(self):
        stations = list(self.stations.iterrows())
        channel = self.config['wavefield_channel']
        if channel == "all":
            channels = ['E', 'N', 'Z']
        else:
            channels = [channel]
        for i, station in stations[self.rank: len(self.stations): self.size]:
            for cha in channels:
                self.function(station, channel=cha)

        if self.rank == 0:
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
        self.npts = int(self.config['wavefield_duration'] * self.Fs)
        self.ntraces = self.sourcegrid.shape[-1]
        self.data_quantity = self.config['synt_data']

        if self.config['wavefield_domain'] == 'fourier':
            self.fdomain = True
            if self.npts % 2 == 1:
                self.npad = 2 * self.npts - 2
            else:
                self.npad = 2 * self.npts
        elif self.config['wavefield_domain'] == 'time':
            self.fdomain = False
            self.npad = next_fast_len(2 * self.npts - 1)
        else:
            raise ValueError('Unknown domain {}.'.format(self.config
                                                         ['wavefield_domain']))
        self.freq = np.fft.rfftfreq(self.npad, d=1.0 / self.Fs)

        # Apply a filter
        if self.config['wavefield_filter'] is not None:
            freq_nyq = self.Fs / 2.0  # Nyquist

            if freq_nyq < self.config['wavefield_filter'][1]:
                warn("Selected upper freq > Nyquist, \
reset to 95\% of Nyquist freq.")
            freq_minres = 1. / self.config['wavefield_duration']
            # lowest resolved
            freq_max = min(0.999 * freq_nyq,
                           self.config['wavefield_filter'][1])
            freq_min = max(freq_minres, self.config['wavefield_filter'][0])

            f0 = freq_min / freq_nyq
            f1 = freq_max / freq_nyq
            self.filter = butter(4, [f0, f1], 'bandpass')
        else:
            self.filter = None

        # if using instaseis: Find and open database
        if self.config['wavefield_type'] == 'instaseis':
            path_to_db = self.config['wavefield_path']
            self.db = instaseis.open_db(path_to_db)
            if self.db.info['length'] < self.npts / self.Fs:
                warn("Resetting wavefield duration to axisem database length.")
                fsrc = instaseis.ForceSource(latitude=0.0,
                                             longitude=0.0, f_r=1.0)
                rec = instaseis.Receiver(latitude=0.0, longitude=0.0)
                test = self.db.get_seismograms(source=fsrc,
                                               receiver=rec,
                                               dt=1. / self.Fs)
                self.npts = test[0].stats.npts

    def green_from_instaseis(self, station, channel):

        # set some parameters
        lat_sta = station['lat']
        lon_sta = station['lon']
        lat_sta = geograph_to_geocent(float(lat_sta))
        lon_sta = float(lon_sta)
        rec = instaseis.Receiver(latitude=lat_sta, longitude=lon_sta)
        point_f = float(self.config['wavefield_point_force'])
        station_id = station['net'] + '.' + station['sta'] + '..MX' + channel

        if self.config['verbose']:
            print(station_id)

        if channel not in ['Z', 'N', 'E']:
            raise ValueError("Unknown channel: %s, choose E, N, Z"
                             % channel)

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
            stats.attrs['npad'] = self.npad
            if self.fdomain:
                stats.attrs['fdomain'] = True
            else:
                stats.attrs['fdomain'] = False

            # DATASET NR 2: Source grid
            f.create_dataset('sourcegrid', data=self.sourcegrid)

            # DATASET Nr 3: Seismograms itself
            if self.fdomain:
                traces = f.create_dataset('data', (self.ntraces,
                                                   self.npts + 1),
                                          dtype=np.complex64)
            else:
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

                trace = values.select(component=channel)[0].data
                if self.filter is not None:
                    trace = lfilter(*self.filter, x=trace)

                if self.fdomain:
                    trace_fd = np.fft.rfft(trace[0: self.npts],
                                           n=self.npad)
                    traces[i, :] = trace_fd
                else:
                    traces[i, :] = trace[0: self.npts]
        return()

    def green_spec_analytic(self, distance_in_m):

        v_phase = self.args.v
        q = self.args.q
        rho = self.args.rho
        freq = self.freq
        w = 2. * pi * freq
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

    def green_analytic(self, station, channel):

        if channel in ["E", "N"]:
            raise ValueError("Analytic Green's function approach is \
invalid for horizontal components; set channel to \"Z\" or use instaseis.")

        # set some parameters
        lat_sta = station['lat']
        lon_sta = station['lon']
        station_id = station['net'] + '.' + station['sta'] + '..MX' + \
            self.config['wavefield_channel']
        if self.config['verbose']:
            print(station_id)

        # initialize the file
        f_out = os.path.join(self.wf_path, station_id + '.h5')

        with h5py.File(f_out, "w", ) as f:

            # DATASET NR 1: STATS
            stats = f.create_dataset('stats', data=(0,))
            stats.attrs['reference_station'] = station['sta']
            stats.attrs['data_quantity'] = self.data_quantity
            stats.attrs['ntraces'] = self.ntraces
            stats.attrs['Fs'] = self.Fs
            stats.attrs['nt'] = int(self.npts)
            stats.attrs['npad'] = self.npad
            if self.fdomain:
                stats.attrs['fdomain'] = True
            else:
                stats.attrs['fdomain'] = False

            # DATASET NR 2: Source grid
            f.create_dataset('sourcegrid', data=self.sourcegrid)

            # DATASET Nr 3: Seismograms itself
            if self.fdomain:
                f.create_dataset('data', (self.ntraces, self.npts),
                                 dtype=np.complex)
            else:
                f.create_dataset('data', (self.ntraces, self.npts),
                                 dtype=np.float)

            # loop over source locations
            for i in range(self.ntraces):
                lat = self.sourcegrid[1, i]
                lon = self.sourcegrid[0, i]
                r = gps2dist_azimuth(lat, lon, lat_sta, lon_sta)[0]

                g_fd = self.green_spec_analytic(r)

                # transform back to time
                s = np.fft.irfft(g_fd, n=self.npad)[0: self.npts]

                # apply a filter if asked for
                if self.filter is not None:
                        trace = lfilter(*self.filter, x=s)

                if self.fdomain:
                    f['data'][i, :] = np.fft.rfft(trace, n=self.npad)
                else:
                    f['data'][i, :] = trace
                # flush
                    f.flush()

            # close output file
            f.close()
        return()
