# from mpi4py import MPI
import numpy as np
import os
import yaml
from glob import glob
from math import ceil
from obspy import Trace
from warnings import warn
from noisi_v1 import NoiseSource, WaveField
from obspy.signal.invsim import cosine_taper
from noisi_v1.util.windows import my_centered
from noisi_v1.util.geo import geograph_to_geocent
from noisi_v1.util.corr_pairs import define_correlationpairs
from noisi_v1.util.corr_pairs import rem_fin_prs, rem_no_obs
try:
    import instaseis
except ImportError:
    pass
# prepare embarrassingly parallel run:
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()


class config_params(object):
    """collection of input parameters"""

    def __init__(self, args, comm, size, rank):
        self.args = args
        source_configfile = os.path.join(self.args.source_model,
                                         'source_config.yml')
        self.step = int(self.args.step)
        self.steplengthrun = self.args.steplengthrun
        self.ignore_network = self.args.ignore_network

        # get configuration
        self.source_config = yaml.safe_load(open(source_configfile))
        self.config = yaml.safe_load(open(os.path.join(self.source_config
                                                       ['project_path'],
                                                       'config.yml')))
        self.obs_only = self.source_config['model_observed_only']
        self.channel = 'MX' + self.config['wavefield_channel']
        self.auto_corr = self.source_config['get_auto_corr']
        with open(os.path.join(self.args.source_model,
                  'measr_config.yml')) as mconf:
            self.measr_config = yaml.safe_load(mconf)
        # Figure out how many frequency bands are measured
        bandpass = self.measr_config['bandpass']
        if bandpass is None:
            self.filtcnt = 1
        elif type(bandpass) == list:
            if type(bandpass[0]) != list:
                self.filtcnt = 1
            else:
                self.filtcnt = len(bandpass)


def add_input_files(cp, all_conf, insta=False):

    inf1 = cp[0].split()
    inf2 = cp[1].split()

    channel = all_conf.config['wavefield_channel']

    # station names
    if all_conf.ignore_network:
        sta1 = "*.{}..??{}".format(*(inf1[1:2] + [channel]))
        sta2 = "*.{}..??{}".format(*(inf2[1:2] + [channel]))
    else:
        sta1 = "{}.{}..??{}".format(*(inf1[0:2] + [channel]))
        sta2 = "{}.{}..??{}".format(*(inf2[0:2] + [channel]))

    # Wavefield files
    if not insta:

        dir = os.path.join(all_conf.config['project_path'], 'greens')
        wf1 = glob(os.path.join(dir, sta1 + '.h5'))[0]
        wf2 = glob(os.path.join(dir, sta2 + '.h5'))[0]
    else:
        # need to return two receiver coordinate pairs. For buried sensors,
        # depth could be used but no elevation is possible.
        # so maybe keep everything at 0 m?
        # lists of information directly from the stations.txt file.
        wf1 = inf1
        wf2 = inf2

    return(wf1, wf2)


def define_correlation_tasks(all_conf, comm, size, rank):

    p = define_correlationpairs(all_conf.source_config
                                ['project_path'],
                                all_conf.auto_corr)
    if rank == 0 and all_conf.config['verbose']:
        print('Nr all possible correlation pairs %g ' % len(p))

    # Remove pairs for which no observation is available
    obs_only = all_conf.source_config['model_observed_only']
    if obs_only:
        if all_conf.steplengthrun:
            directory = os.path.join(all_conf.source_config['source_path'],
                                     'observed_correlations_slt')
        else:
            directory = os.path.join(all_conf.source_config['source_path'],
                                     'observed_correlations')
        if rank == 0:
            # split p into size lists for comm.scatter()
            p_split = np.array_split(p, size)
            p_split = [k.tolist() for k in p_split]
        else:
            p_split = None

        # scatter p_split to ranks
        p_split = comm.scatter(p_split, root=0)
        p_split = rem_no_obs(p_split, all_conf.source_config,
                             directory=directory)

        # gather all on rank 0
        p_new = comm.gather(list(p_split), root=0)

        # put all back into one array p
        if rank == 0:
            p = [i for j in p_new for i in j]

        # broadcast p to all ranks
        p = comm.bcast(p, root=0)
        if rank == 0 and all_conf.config['verbose']:
            print('Nr correlation pairs after checking available observ. %g '
                  % len(p))

    # Remove pairs that have already been calculated
    p = rem_fin_prs(p, all_conf.source_config, all_conf.step)
    if rank == 0 and all_conf.config['verbose']:
        print('Nr correlation pairs after checking already calculated ones %g'
              % len(p))
        print(16 * '*')

    # The assignment of station pairs should be such that one core has as
    # many occurrences of the same station as possible;
    # this will prevent that many processes try to read from the same hdf5
    # file all at once.
    num_pairs = int(ceil(float(len(p)) / float(size)))

    p_p = p[rank * num_pairs: rank * num_pairs + num_pairs]

    return(p_p, num_pairs, len(p))


def add_output_file(cp, all_conf):

    id1 = cp[0].split()[0] + cp[0].split()[1]
    id2 = cp[1].split()[0] + cp[1].split()[1]

    if id1 < id2:
        inf1 = cp[0].split()
        inf2 = cp[1].split()
    else:
        inf2 = cp[0].split()
        inf1 = cp[1].split()

    sta1 = "{}.{}..{}".format(*(inf1[0:2] + [all_conf.channel]))
    sta2 = "{}.{}..{}".format(*(inf2[0:2] + [all_conf.channel]))

    corr_trace_name = "{}--{}.sac".format(sta1, sta2)
    corr_trace_name = os.path.join(all_conf.source_config['source_path'],
                                   'iteration_' + str(all_conf.step), 'corr',
                                   corr_trace_name)
    return corr_trace_name


def get_ns(all_conf, insta=False):
    # Nr of time steps in traces
    if insta:
        # get path to instaseis db
        dbpath = all_conf.config['wavefield_path']

        # open
        db = instaseis.open_db(dbpath)
        # get a test seismogram to determine...
        stest = db.get_seismograms(source=instaseis.ForceSource(latitude=0.0,
                                                                longitude=0.),
                                   receiver=instaseis.Receiver(latitude=10.,
                                                               longitude=0.),
                                   dt=1. / all_conf.config
                                   ['wavefield_sampling_rate'])[0]
        nt = stest.stats.npts
        Fs = stest.stats.sampling_rate
    else:
        any_wavefield = glob(os.path.join(all_conf.config['project_path'],
                                          'greens', '*.h5'))[-1]
        with WaveField(any_wavefield) as wf1:
            nt = int(wf1.stats['nt'])
            Fs = round(wf1.stats['Fs'], 8)
            n = wf1.stats['npad']
    # # Necessary length of zero padding
    # # for carrying out frequency domain correlations/convolutions
    # n = next_fast_len(2 * nt - 1)

    # Number of time steps for synthetic correlation
    n_lag = int(all_conf.source_config['max_lag'] * Fs)
    if nt - 2 * n_lag <= 0:
        n_lag_old = n_lag
        n_lag = nt // 2
        warn('Resetting maximum lag to %g seconds:\
 Synthetics are too short for %g seconds.' % (n_lag / Fs, n_lag_old / Fs))

    n_corr = 2 * n_lag + 1

    return nt, n, n_corr, Fs


def compute_correlation(input_files, all_conf, nsrc, all_ns, taper,
                        insta=False):
    """
    Compute noise cross-correlations from two .h5 'wavefield' files.
    Noise source distribution and spectrum is given by starting_model.h5
    It is assumed that noise sources are delta-correlated in space.

    Metainformation: Include the reference station names for both stations
    from wavefield files, if possible. Do not include geographic information
    from .csv file as this might be error-prone. Just add the geographic
    info later if needed.
    """

    wf1, wf2 = input_files
    ntime, n, n_corr, Fs = all_ns
    ntraces = nsrc.src_loc[0].shape[0]
    correlation = np.zeros(n_corr)

    if insta:
        # open database
        dbpath = all_conf.config['wavefield_path']

        # open
        db = instaseis.open_db(dbpath)
        # get receiver locations
        station1 = wf1[0]
        station2 = wf2[0]
        lat1 = geograph_to_geocent(float(wf1[2]))
        lon1 = float(wf1[3])
        rec1 = instaseis.Receiver(latitude=lat1, longitude=lon1)
        lat2 = geograph_to_geocent(float(wf2[2]))
        lon2 = float(wf2[3])
        rec2 = instaseis.Receiver(latitude=lat2, longitude=lon2)
    else:
        wf1 = WaveField(wf1)
        wf2 = WaveField(wf2)
        station1 = wf1.stats['reference_station']
        station2 = wf2.stats['reference_station']

        # Make sure all is consistent
        if False in (wf1.sourcegrid[1, 0:10] == wf2.sourcegrid[1, 0:10]):
            raise ValueError("Wave fields not consistent.")

        if False in (wf1.sourcegrid[1, -10:] == wf2.sourcegrid[1, -10:]):
            raise ValueError("Wave fields not consistent.")

        if False in (wf1.sourcegrid[0, -10:] == nsrc.src_loc[0, -10:]):
            raise ValueError("Wave field and source not consistent.")

    # Loop over source locations
    print_each_n = max(5, round(max(ntraces // 5, 1), -1))
    for i in range(ntraces):

        # noise source spectrum at this location
        S = nsrc.get_spect(i)

        if S.sum() == 0.:
            # If amplitude is 0, continue. (Spectrum has 0 phase anyway.)
            continue

        if insta:
            # get source locations
            lat_src = geograph_to_geocent(nsrc.src_loc[1, i])
            lon_src = nsrc.src_loc[0, i]
            fsrc = instaseis.ForceSource(latitude=lat_src,
                                         longitude=lon_src,
                                         f_r=1.e12)
            Fs = all_conf.config['wavefield_sampling_rate']
            s1 = db.get_seismograms(source=fsrc, receiver=rec1,
                                    dt=1. / Fs)[0].data * taper
            s2 = db.get_seismograms(source=fsrc, receiver=rec2,
                                    dt=1. / Fs)[0].data * taper
            s1 = np.ascontiguousarray(s1)
            s2 = np.ascontiguousarray(s2)
            spec1 = np.fft.rfft(s1, n)
            spec2 = np.fft.rfft(s2, n)

        else:
            if not wf1.fdomain:
                # read Green's functions
                s1 = np.ascontiguousarray(wf1.data[i, :] * taper)
                s2 = np.ascontiguousarray(wf2.data[i, :] * taper)
                # Fourier transform for greater ease of convolution
                spec1 = np.fft.rfft(s1, n)
                spec2 = np.fft.rfft(s2, n)
            else:
                spec1 = np.ascontiguousarray(wf1.data[i, :])
                spec2 = np.ascontiguousarray(wf2.data[i, :])

        # convolve G1G2
        g1g2_tr = np.multiply(np.conjugate(spec1), spec2)

        # convolve noise source
        c = np.multiply(g1g2_tr, S)

        # transform back
        correlation += my_centered(np.fft.fftshift(np.fft.irfft(c, n)),
                                   n_corr) * nsrc.surf_area[i]
        # occasional info
        if i % print_each_n == 0 and all_conf.config['verbose']:
            print("Finished {} of {} source locations.".format(i, ntraces))
# end of loop over all source locations #######################################
    return(correlation, station1, station2)


def add_metadata_and_write(correlation, sta1, sta2, output_file, Fs):
    # save output
    trace = Trace()
    trace.stats.sampling_rate = Fs
    trace.data = correlation
    # try to add some meta data
    try:
        trace.stats.station = sta1.split('.')[1]
        trace.stats.network = sta1.split('.')[0]
        trace.stats.location = sta1.split('.')[2]
        trace.stats.channel = sta1.split('.')[3]
        trace.stats.sac = {}
        trace.stats.sac['kuser0'] = sta2.split('.')[1]
        trace.stats.sac['kuser1'] = sta2.split('.')[0]
        trace.stats.sac['kuser2'] = sta2.split('.')[2]
        trace.stats.sac['kevnm'] = sta2.split('.')[3]
    except (KeyError, IndexError):
        pass

    trace.write(filename=output_file, format='SAC')
    return()


def run_corr(args, comm, size, rank):

    all_conf = config_params(args, comm, size, rank)
    # Distributing the tasks
    correlation_tasks, n_p_p, n_p = define_correlation_tasks(all_conf,
                                                             comm, size, rank)
    if len(correlation_tasks) == 0:
        return()
    if all_conf.config['verbose']:
        print('Rank number %g' % rank)
        print('working on pair nr. %g to %g of %g.' % (rank * n_p_p,
                                                       rank * n_p_p +
                                                       n_p_p, n_p))

    # Current model for the noise source
    nsrc = os.path.join(all_conf.source_config['project_path'],
                        all_conf.source_config['source_name'],
                        'iteration_' + str(all_conf.step),
                        'starting_model.h5')

    # Smart numbers
    all_ns = get_ns(all_conf)  # ntime, n, n_corr, Fs

    # use a one-sided taper: The seismogram probably has a non-zero end,
    # being cut off wherever the solver stopped running.
    taper = cosine_taper(all_ns[0], p=0.01)
    taper[0: all_ns[0] // 2] = 1.0

    with NoiseSource(nsrc) as nsrc:
        for cp in correlation_tasks:
            try:
                input_files = add_input_files(cp, all_conf)

                output_file = add_output_file(cp, all_conf)
            except (IndexError, FileNotFoundError):
                if all_conf.config['verbose']:
                    print('Could not determine correlation for: %s\
    \nCheck if wavefield .h5 file is available.' % cp)
                continue

            correlation, sta1, sta2 = compute_correlation(input_files,
                                                          all_conf, nsrc,
                                                          all_ns, taper)
            add_metadata_and_write(correlation, sta1, sta2,
                                   output_file, all_ns[3])

    return()
