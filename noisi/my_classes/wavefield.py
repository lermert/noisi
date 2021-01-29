from __future__ import print_function
import numpy as np
import h5py

from noisi.util import filter
try:
    from scipy.signal import sosfilt
except ImportError:
    from obspy.signal._sosfilt import _sosfilt as sosfilt
from scipy.fftpack import next_fast_len
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import integer_decimation
from warnings import warn
from os.path import splitext


class WaveField(object):
    """
    Object to handle database of stored wavefields.
    Methods to work on wavefields stored in an hdf5 file.
    """

    def __init__(self, file, sourcegrid=None, w='r', preload=False):
        self.w = w
        self.data = {}
        self.preload = preload

        try:
            self.file = h5py.File(file, self.w)
        except IOError:
            msg = 'Unable to open input file ' + file
            raise IOError(msg)

        self.stats = dict(self.file['stats'].attrs)
        try:
            self.fdomain = self.stats['fdomain']
        except KeyError:
            self.fdomain = False
        self.sourcegrid = self.file['sourcegrid']
        self.datakeys = []

        if 'npad' not in self.stats:
            if self.fdomain:
                self.stats['npad'] = 2 * self.stats['nt'] - 2
            else:
                self.stats['npad'] = next_fast_len(2 * self.stats['nt'] - 1)

        try:
            if self.preload:
                self.data["data"] = self.file["data"][:]
            else:
                self.data['data'] = self.file['data']
        except KeyError:
            pass

        # traction source Green's functions
        try:
            if self.preload:
                self.data['data_fz'] = self.file['data_fz'][:]
            else:
                self.data['data_fz'] = self.file['data_fz']
        except KeyError:
            pass
        
        try:
            if self.preload:
                self.data['data_fn'] = self.file['data_fn'][:]
            else:
                self.data['data_fn'] = self.file['data_fn']

        except KeyError:
            pass

        try:
            if self.preload:
                self.data['data_fe'] = self.file['data_fe'][:]
            else:
                self.data['data_fe'] = self.file['data_fe']

        except KeyError:
            pass

        if len(self.data) == 0:
            raise ValueError("No usable data in file.")

        if self.fdomain:
            self.freq = np.fft.rfftfreq(self.stats['npad'],
                                        d=1. / self.stats['Fs'])


    def get_green(self, ix=None, by_index=True):

        if by_index:
            data_out = np.empty((len(self.data), self.stats['nt']))
            for ix_data, key in enumerate(self.data.keys()):
                data_out[ix_data, :] = self.data[key][ix, :]
            return(data_out)
        else:
            raise NotImplementedError("Not yet implemented.")

    def copy_setup(self, newfile, nt=None):

        # Create new file
        file = h5py.File(newfile, 'w-')

        # Copy metadata
        file.create_dataset('stats', data=(0, ))
        for (key, value) in self.stats.items():
            file['stats'].attrs[key] = value

        # Ensure that nt is kept as requested
        if nt is not None and nt != self.stats['nt']:
            file['stats'].attrs['nt'] = nt

        file.create_dataset('sourcegrid', data=self.sourcegrid[:].copy())

        for key, values in self.data.items():
            file.create_dataset(key, values.shape, dtype=np.float32)

        print('Copied setup of ' + self.file.filename)
        file.close()

        return(WaveField(newfile, w="a", preload=self.preload))

    def truncate(self, newfile, truncate_after_seconds):

        if self.fdomain:
            raise NotImplementedError()

        nt_new = int(round(truncate_after_seconds * self.stats['Fs']))
        with self.copy_setup(newfile, nt=nt_new) as wf:
            for key, values in self.data.items():
                for i in range(self.stats['ntraces']):
                    wf.data[key][i, :] = values[i, 0:nt_new].copy()

        return()

    def filter_all(self, type, overwrite=False, zerophase=True,
                   outfile=None, **kwargs):

        if self.fdomain:
            raise NotImplementedError('Not implemented yet, filter beforehand')

        if type == 'bandpass':
            sos = filter.bandpass(df=self.stats['Fs'], **kwargs)
        elif type == 'lowpass':
            sos = filter.lowpass(df=self.stats['Fs'], **kwargs)
        elif type == 'highpass':
            sos = filter.highpass(df=self.stats['Fs'], **kwargs)
        else:
            msg = 'Filter %s is not implemented, implemented filters:\
bandpass, highpass,lowpass' % type
            raise ValueError(msg)

        if not overwrite:
            # Create a new hdf5 file of the same shape
            newfile = self.copy_setup(newfile=outfile)
        else:
            # Call self.file newfile
            newfile = self

        for key, values in self.data.items():
            for i in range(self.stats['ntraces']):
                # Filter each trace
                if zerophase:
                    firstpass = sosfilt(sos, values[i, :])
                    # Read in any case from self.data
                    newfile.data[key][i, :] =\
                        sosfilt(sos, firstpass[::-1])[::-1]
                    # then assign to newfile, which might be self.file
                else:
                    newfile.data[i, :] = sosfilt(sos, values[i, :])
            if not overwrite:
                print('Processed traces written to file %s, file closed. \
Reopen the file to read / modify.' % newfile.file.filename)

            newfile.file.close()

    def decimate(self, decimation_factor, outfile, taper_width=0.005):
        """
        Decimate the wavefield and save to a new file
        """
        if self.fdomain:
            raise NotImplementedError()

        fs_old = self.stats['Fs']
        freq = self.stats['Fs'] * 0.4 / float(decimation_factor)

        # Get filter coeff
        sos = filter.cheby2_lowpass(fs_old, freq)

        # figure out new length
        temp_trace = integer_decimation(self[self.data.keys()[0]].data[0, :],
                                        decimation_factor)
        n = len(temp_trace)

        # Get taper
        # The default taper is very narrow,
        # because it is expected that the traces are very long.
        taper = cosine_taper(self.stats['nt'], p=taper_width)

        # Need a new file, because the length changes.
        with self.copy_setup(newfile=outfile, nt=n) as newfile:

            for key, values in self.data.items():
                for i in range(self.stats['ntraces']):
                    temp_trace = sosfilt(sos, taper * values[i, :])
                    newfile.data[key][i, :] =\
                        integer_decimation(temp_trace, decimation_factor)

            newfile.stats['Fs'] = fs_old / float(decimation_factor)

    def get_snapshot(self, t, resolution=1):

        t_sample = int(round(self.stats['Fs'] * t))
        if t_sample >= self.stats['nt']:
            warn('Requested sample is out of bounds,\
resetting to last sample.')
            t_sample = self.stats['nt'] - 1

        snapshots = {}

        for key, values in self.data.items():
            if resolution == 1:
                if self.fdomain:
                    snapshots[key] = np.zeros(self.stats['ntraces'])
                    for i in range(self.stats['ntraces']):
                        snapshots[key][i] =\
                            np.trapz(values[i, :] * np.exp(self.freq *
                                     np.pi * 2. * 1.j * t_sample),
                                     dx=self.stats['Fs']).real
                else:
                    snapshots[key] = values[:, t_sample]
            else:
                if self.fdomain:
                    snapshots[key] = np.zeros(values[0::resolution, 0].shape)
                    for i in range(len(snapshots[key])):
                        snapshots[key][i] =\
                            np.trapz(values[i, :] * np.exp(self.freq *
                                     np.pi * 2. * 1.j * t_sample),
                                     dx=1. / self.stats['Fs']).real
                else:
                    snapshots[key] = values[0::resolution, t_sample]
        return snapshots

    def plot_snapshot(self, t, resolution=1, **kwargs):

        if 'plot' not in globals():
            print("Cannot plot, is cartopy installed?")
            return()

        if self.sourcegrid is None:
            msg = 'Must have a source grid to plot a snapshot.'
            raise ValueError(msg)

        map_x = self.sourcegrid[0][0::resolution]
        map_y = self.sourcegrid[1][0::resolution]

        if self.stats['data_quantity'] == 'DIS':
            quant_unit = 'Displacement (m)'
        elif self.stats['data_quantity'] == 'VEL':
            quant_unit = 'Velocity (m/s)'
        elif self.stats['data_quantity'] == 'ACC':
            quant_unit = 'Acceleration (m/s\u00B2)'

        snapshots = self.get_snapshot(t, resolution=resolution)
        for key, vals in snapshots.items():
            src = key[-1]
            if 'outfile' in kwargs:
                outf = splitext(kwargs['outfile'])[0]
                kwargs['outfile'] = outf + '.' + key + '.png'
            plot.plot_grid(map_x, map_y, vals,
                           title='Wave field of %s-comp. \
source after %g seconds' % (src, t),
                           quant_unit=quant_unit,
                           **kwargs)

    def update_stats(self):

        if self.w != 'r':
            print('Updating stats...')
            self.file['stats'].attrs['ntraces'] =\
                self.data[self.data.keys()[0]].shape[0]
            self.file['stats'].attrs['nt'] = \
                self.data[self.data.keys()[0]].shape[1]
            if 'stats' not in self.file.keys():
                self.file.create_dataset('stats', data=(0,))
            for (key, value) in self.stats.items():
                self.file['stats'].attrs[key] = value

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):

        self.update_stats()
        self.file.close()

