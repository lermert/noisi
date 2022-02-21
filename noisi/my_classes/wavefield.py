"""
Class for handling Green's function library in noisi
:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import print_function
import numpy as np
import h5py
try:
    from noisi.util import plot
except ImportError:
    print("Plotting unavailable (is cartopy installed?)", end='\n')
    pass
from noisi.util import filter
try:
    from scipy.signal import sosfilt
except ImportError:
    from obspy.signal._sosfilt import _sosfilt as sosfilt

from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import integer_decimation
from scipy.fftpack import next_fast_len
from warnings import warn


class WaveField(object):
    """
    Object to handle database of stored wavefields.
    Methods to work on wavefields stored in an hdf5 file.
    """

    def __init__(self, file, sourcegrid=None, w='r'):

        self.w = w

        try:
            self.file = h5py.File(file, self.w)
        except IOError:
            msg = 'Unable to open input file ' + file
            raise IOError(msg)

        self.stats = dict(self.file['stats'].attrs)
        self.fdomain = self.stats['fdomain']
        self.sourcegrid = self.file['sourcegrid']
        
        try:
            self.data = self.file['data']
        except KeyError:
            self.data = self.file['data_z']
        if self.fdomain:
            self.freq = np.fft.rfftfreq(self.stats['npad'],
                                        d=1. / self.stats['Fs'])
        if 'npad' not in self.stats:
            if self.fdomain:
                self.stats['npad'] = 2 * self.stats['nt'] - 2
            else:
                self.stats['npad'] = next_fast_len(2 * self.stats['nt'] - 1)

    def get_green(self, ix=None, by_index=True):

        if by_index:
            return(self.data[ix, :])
        else:
            raise NotImplementedError("Not implemented.")

    def copy_setup(self, newfile, nt=None,
                   ntraces=None, w='r+'):

        # Shape of the new array:
        shape = list(np.shape(self.data))
        if ntraces is not None:
            shape[0] = ntraces
        if nt is not None:
            shape[1] = nt
        shape = tuple(shape)
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
        file.create_dataset('data', shape, dtype=np.float32)

        print('Copied setup of ' + self.file.filename)
        file.close()

        return(WaveField(newfile, w=w))

    def truncate(self, newfile, truncate_after_seconds):

        if self.fdomain:
            raise NotImplementedError()

        nt_new = int(round(truncate_after_seconds * self.stats['Fs']))
        with self.copy_setup(newfile, nt=nt_new) as wf:
            for i in range(self.stats['ntraces']):
                if self.complex:
                    wf.data_i[i, :] = self.data_i[i, 0:nt_new].copy()
                    wf.data_r[i, :] = self.data_r[i, 0:nt_new].copy()
                else:
                    wf.data[i, :] = self.data[i, 0:nt_new].copy()

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

        for i in range(self.stats['ntraces']):
            # Filter each trace
            if zerophase:
                firstpass = sosfilt(sos, self.data[i, :])
                # Read in any case from self.data
                newfile.data[i, :] = sosfilt(sos, firstpass[::-1])[::-1]
                # then assign to newfile, which might be self.file
            else:
                newfile.data[i, :] = sosfilt(sos, self.data[i, :])
        if not overwrite:
            print('Processed traces written to file %s, file closed, \
reopen to read / modify.' % newfile.file.filename)

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
        temp_trace = integer_decimation(self.data[0, :], decimation_factor)
        n = len(temp_trace)

        # Get taper
        # The default taper is very narrow,
        # because it is expected that the traces are very long.
        taper = cosine_taper(self.stats['nt'], p=taper_width)

        # Need a new file, because the length changes.
        with self.copy_setup(newfile=outfile, nt=n) as newfile:

            for i in range(self.stats['ntraces']):
                temp_trace = sosfilt(sos, taper * self.data[i, :])
                newfile.data[i, :] = integer_decimation(temp_trace,
                                                        decimation_factor)

            newfile.stats['Fs'] = fs_old / float(decimation_factor)

    def get_snapshot(self, t, resolution=1):

        t_sample = int(round(self.stats['Fs'] * t))
        if t_sample >= np.shape(self.data)[1]:
            warn('Requested sample is out of bounds,\
resetting to last sample.')
            t_sample = np.shape(self.data)[1] - 1

        if resolution == 1:
            if self.fdomain:
                snapshot = np.zeros(self.stats['ntraces'])
                for i in range(self.stats['ntraces']):
                    snapshot[i] = np.trapz(self.data[i, :] *
                                           np.exp(self.freq *
                                           np.pi * 2. * 1.j * t_sample),
                                           dx=self.stats['Fs']).real
            else:
                snapshot = self.data[:, t_sample]
        else:
            if self.fdomain:
                snapshot = np.zeros(self.data[0::resolution, 0].shape)
                for i in range(len(snapshot)):
                    snapshot[i] = np.trapz(self.data[i, :] *
                                           np.exp(self.freq *
                                           np.pi * 2. * 1.j * t_sample),
                                           dx=1. / self.stats['Fs']).real
            else:
                snapshot = self.data[0::resolution, t_sample]
        return snapshot

    def plot_snapshot(self, t, resolution=1, **kwargs):

        if 'plot' not in globals():
            print("Cannot plot, is cartopy installed?")
            return()

        if self.sourcegrid is None:
            msg = 'Must have a source grid to plot a snapshot.'
            raise ValueError(msg)

        map_x = self.sourcegrid[0][0::resolution]
        map_y = self.sourcegrid[1][0::resolution]

        if type(self.stats['data_quantity']) != str:
            data_quantity = self.stats['data_quantity'].decode()
        else:
            data_quantity = self.stats['data_quantity']
        if  data_quantity == 'DIS':
            quant_unit = 'Displacement (m)'
        elif data_quantity == 'VEL':
            quant_unit = 'Velocity (m/s)'
        elif data_quantity == 'ACC':
            quant_unit = 'Acceleration (m/s^2)'
        else:
            print("unknown data quantity")
            quant_unit = "Stuff"

        plot.plot_grid(map_x, map_y,
                       self.get_snapshot(t, resolution=resolution),
                       title='Discretized wave field after %g seconds' % t,
                       quant_unit=quant_unit,
                       **kwargs)

    def update_stats(self):

        if self.w != 'r':
            print('Updating stats...')
            self.file['stats'].attrs['ntraces'] = self.data.shape[0]
            self.file['stats'].attrs['nt'] = self.data.shape[-1]

            if 'stats' not in self.file.keys():
                self.file.create_dataset('stats', data=(0,))
            for (key, value) in self.stats.items():
                self.file['stats'].attrs[key] = value

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):

        self.update_stats()
        self.file.close()
