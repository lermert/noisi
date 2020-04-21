import numpy as np
from obspy.signal.invsim import cosine_taper
import h5py
import yaml
import time
from glob import glob
import os
import io
import errno
from noisi import WaveField
from noisi.util.geo import is_land, geographical_distances
from noisi.util.geo import get_spherical_surface_elements
try:
    from noisi.util.plot import plot_grid
    create_plot = True
except ImportError:
    create_plot = False
    pass
import matplotlib.pyplot as plt
from math import pi, sqrt
from warnings import warn
import pprint


class source_setup(object):

    def __init__(self, args, comm, size, rank):

        if not args.new_model:
            self.setup_source_startingmodel(args)
        else:
            self.initialize_source(args)

    def initialize_source(self, args):
        source_model = args.source_model
        project_path = os.path.dirname(source_model)
        noisi_path = os.path.abspath(os.path.dirname(
                                     os.path.dirname(__file__)))
        config_filename = os.path.join(project_path, 'config.yml')

        if not os.path.exists(config_filename):
            raise FileNotFoundError(errno.ENOENT,
                                    os.strerror(errno.ENOENT) +
                                    "\nRun setup_project first.",
                                    config_filename)

        # set up the directory structure:
        os.mkdir(source_model)
        os.mkdir(os.path.join(source_model, 'observed_correlations'))
        for d in ['adjt', 'corr', 'kern']:
            os.makedirs(os.path.join(source_model, 'iteration_0', d))

        # set up the source model configuration file
        with io.open(os.path.join(noisi_path,
                                  'config', 'source_config.yml'), 'r') as fh:
            conf = yaml.safe_load(fh)
            conf['date_created'] = str(time.strftime("%Y.%m.%d"))
            conf['project_name'] = os.path.basename(project_path)
            conf['project_path'] = os.path.abspath(project_path)
            conf['source_name'] = os.path.basename(source_model)
            conf['source_path'] = os.path.abspath(source_model)
            conf['source_setup_file'] = os.path.join(conf['source_path'],
                                              'source_setup_parameters.yml')

        with io.open(os.path.join(noisi_path,
                                  'config',
                                  'source_config_comments.txt'), 'r') as fh:
            comments = fh.read()

        with io.open(os.path.join(source_model,
                                  'source_config.yml'), 'w') as fh:
            cf = yaml.safe_dump(conf, sort_keys=False, indent=4)
            fh.write(cf)
            fh.write(comments)

        # set up the measurements configuration file
        with io.open(os.path.join(noisi_path,
                                  'config', 'measr_config.yml'), 'r') as fh:
            conf = yaml.safe_load(fh)
            conf['date_created'] = str(time.strftime("%Y.%m.%d"))
        with io.open(os.path.join(noisi_path,
                                  'config',
                                  'measr_config_comments.txt'), 'r') as fh:
            comments = fh.read()

        with io.open(os.path.join(source_model,
                                  'measr_config.yml'), 'w') as fh:
            cf = yaml.safe_dump(conf, sort_keys=False, indent=4)
            fh.write(cf)
            fh.write(comments)

        # set up the measurements configuration file
        with io.open(os.path.join(noisi_path,
                                  'config',
                                  'source_setup_parameters.yml'), 'r') as fh:
            setup = yaml.safe_load(fh)

        with io.open(os.path.join(source_model,
                                  'source_setup_parameters.yml'), 'w') as fh:
            stup = yaml.safe_dump(setup, sort_keys=False, indent=4)
            fh.write(stup)

        os.system('cp ' +
                  os.path.join(noisi_path, 'config', 'stationlist.csv ') +
                  source_model)

        print("Copied default source_config.yml, source_setup_parameters.yml \
and measr_config.yml to source model directory, please edit and rerun.")
        return()

    def setup_source_startingmodel(self, args):

        # plotting:
        colors = ['purple', 'g', 'b', 'orange']
        colors_cmaps = [plt.cm.Purples, plt.cm.Greens, plt.cm.Blues,
                        plt.cm.Oranges]
        print("Setting up source starting model.", end="\n")
        with io.open(os.path.join(args.source_model,
                                  'source_config.yml'), 'r') as fh:
            source_conf = yaml.safe_load(fh)

        with io.open(os.path.join(source_conf['project_path'],
                                  'config.yml'), 'r') as fh:
            conf = yaml.safe_load(fh)

        with io.open(source_conf['source_setup_file'], 'r') as fh:
            parameter_sets = yaml.safe_load(fh)
            if conf['verbose']:
                print("The following input parameters are used:", end="\n")
                pp = pprint.PrettyPrinter()
                pp.pprint(parameter_sets)

        # load the source locations of the grid
        grd = np.load(os.path.join(conf['project_path'],
                                   'sourcegrid.npy'))

        # add the approximate spherical surface elements
        if grd.shape[-1] < 50000:
            surf_el = get_spherical_surface_elements(grd[0], grd[1])
        else:
            warn('Large grid; surface element computation slow. Using \
approximate surface elements.')
            surf_el = np.ones(grd.shape[-1]) * conf['grid_dx_in_m'] ** 2

        # get the relevant array sizes
        wfs = glob(os.path.join(conf['project_path'], 'greens', '*.h5'))
        if wfs != []:
            if conf['verbose']:
                print('Found wavefield stats.')
            else:
                pass
        else:
            raise FileNotFoundError('No wavefield database found. Run \
precompute_wavefield first.')
        with WaveField(wfs[0]) as wf:
            df = wf.stats['Fs']
            n = wf.stats['npad']
        freq = np.fft.rfftfreq(n, d=1. / df)
        n_distr = len(parameter_sets)
        coeffs = np.zeros((grd.shape[-1], n_distr))
        spectra = np.zeros((n_distr, len(freq)))

        # fill in the distributions and the spectra
        for i in range(n_distr):
            coeffs[:, i] = self.distribution_from_parameters(grd,
                                                             parameter_sets[i],
                                                             conf['verbose'])

            # plot
            outfile = os.path.join(args.source_model,
                                   'source_starting_model_distr%g.png' % i)
            if create_plot:
                plot_grid(grd[0], grd[1], coeffs[:, i],
                          outfile=outfile, cmap=colors_cmaps[i%len(colors_cmaps)],
                          sequential=True, normalize=False,
                          quant_unit='Spatial weight (-)',
                          axislabelpad=-0.1,
                          size=10)

            spectra[i, :] = self.spectrum_from_parameters(freq,
                                                          parameter_sets[i])

        # plotting the spectra
        # plotting is not necessarily done to make sure code runs on clusters
        if create_plot:
            fig1 = plt.figure()
            ax = fig1.add_subplot('111')
            for i in range(n_distr):
                ax.plot(freq, spectra[i, :] / spectra.max(),
                        color=colors[i%len(colors_cmaps)])

            ax.set_xlabel('Frequency / Nyquist Frequency')
            plt.xticks([0, freq.max() * 0.25, freq.max() * 0.5,
                       freq.max() * 0.75, freq.max()],
                       ['0', '0.25', '0.5', '0.75', '1'])
            ax.set_ylabel('Rel. PSD norm. to strongest spectrum (-)')
            fig1.savefig(os.path.join(args.source_model,
                                      'source_starting_model_spectra.png'))

        # Save to an hdf5 file
        with h5py.File(os.path.join(args.source_model, 'iteration_0',
                                    'starting_model.h5'), 'w') as fh:
            fh.create_dataset('coordinates', data=grd)
            fh.create_dataset('frequencies', data=freq)
            fh.create_dataset('model', data=coeffs.astype(np.float))
            fh.create_dataset('spectral_basis',
                              data=spectra.astype(np.float))
            fh.create_dataset('surface_areas',
                              data=surf_el.astype(np.float))

        # Save to an hdf5 file
        with h5py.File(os.path.join(args.source_model,
                                    'spectral_model.h5'), 'w') as fh:
            uniform_spatial = np.ones(coeffs.shape) * 1.0
            fh.create_dataset('coordinates', data=grd)
            fh.create_dataset('frequencies', data=freq)
            fh.create_dataset('model', data=uniform_spatial.astype(np.float))
            fh.create_dataset('spectral_basis',
                              data=spectra.astype(np.float))
            fh.create_dataset('surface_areas',
                              data=surf_el.astype(np.float))

    def distribution_from_parameters(self, grd, parameters, verbose=False):

        if parameters['distribution'] == 'homogeneous':
            if verbose:
                print('Adding homogeneous distribution')
            distribution = np.ones(grd.shape[-1])
            return(float(parameters['weight']) * distribution)

        elif parameters['distribution'] == 'ocean':
            if verbose:
                print('Adding ocean-only distribution')
            is_ocean = np.abs(is_land(grd[0], grd[1]) - 1.)
            return(float(parameters['weight']) * is_ocean)

        elif parameters['distribution'] == 'gaussian_blob':
            if verbose:
                print('Adding gaussian blob')
            dist = geographical_distances(grd,
                                          parameters['center_latlon']) / 1000.
            sigma_km = parameters['sigma_m'] / 1000.
            blob = np.exp(-(dist ** 2) / (2 * sigma_km ** 2))
            # normalize for a 2-D Gaussian function
            # important: Use sigma in m because the surface elements are in m
            norm_factor = 1. / ((sigma_km * 1000.) ** 2 * 2. * np.pi)
            blob *= norm_factor
            if parameters['normalize_blob_to_unity']:
                blob /= blob.max()

            if parameters['only_in_the_ocean']:
                is_ocean = np.abs(is_land(grd[0], grd[1]) - 1.)
                blob *= is_ocean

            return(float(parameters['weight']) * blob)

    def spectrum_from_parameters(self, freq, parameters):

        mu = parameters['mean_frequency_Hz']
        sig = parameters['standard_deviation_Hz']
        taper = cosine_taper(len(freq), parameters['taper_percent'] / 100.)
        spec = taper * np.exp(-((freq - mu) ** 2) /
                              (2 * sig ** 2))

        if not parameters['normalize_spectrum_to_unity']:
            spec = spec / (sig * sqrt(2. * pi))

        return(spec)
