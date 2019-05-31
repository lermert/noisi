from mpi4py import MPI
import numpy as np
import os
import yaml
from glob import glob
from math import ceil
from scipy.fftpack import next_fast_len
from obspy import Trace
from warnings import warn
from noisi_v1 import NoiseSource, WaveField
from obspy.signal.invsim import cosine_taper
from noisi_v1.util.windows import my_centered
from noisi_v1.util.geo import geograph_to_geocent
from noisi_v1.util.corr_pairs import define_correlationpairs
from noisi_v1.util.corr_pairs import rem_fin_prs, rem_no_obs
import instaseis


def paths_input(cp, source_conf, step, ignore_network, instaseis=False):

    inf1 = cp[0].split()
    inf2 = cp[1].split()

    conf = yaml.safe_load(open(os.path.join(source_conf['project_path'],
                                            'config.yml')))
    channel = conf['wavefield_channel']

    # station names
    if ignore_network:
        sta1 = "*.{}..??{}".format(*(inf1[1:2] + [channel]))
        sta2 = "*.{}..??{}".format(*(inf2[1:2] + [channel]))
    else:
        sta1 = "{}.{}..??{}".format(*(inf1[0:2] + [channel]))
        sta2 = "{}.{}..??{}".format(*(inf2[0:2] + [channel]))

    # Wavefield files
    if not instaseis:

        dir = os.path.join(conf['project_path'], 'greens')

        wf1 = glob(os.path.join(dir, sta1 + '.h5'))[0]
        wf2 = glob(os.path.join(dir, sta2 + '.h5'))[0]
    else:
        # need to return two receiver coordinate pairs. For buried sensors,
        # depth could be used but no elevation is possible.
        # so maybe keep everything at 0 m?
        # lists of information directly from the stations.txt file.
        wf1 = inf1
        wf2 = inf2

    # Starting model for the noise source
    nsrc = os.path.join(source_conf['project_path'],
                        source_conf['source_name'], 'iteration_' + str(step),
                        'starting_model.h5')
    return(wf1, wf2, nsrc)


def path_output(cp, source_conf, step, channel):

    id1 = cp[0].split()[0] + cp[0].split()[1]
    id2 = cp[1].split()[0] + cp[1].split()[1]

    if id1 < id2:
        inf1 = cp[0].split()
        inf2 = cp[1].split()
    else:
        inf2 = cp[0].split()
        inf1 = cp[1].split()

    sta1 = "{}.{}..{}".format(*(inf1[0:2] + [channel]))
    sta2 = "{}.{}..{}".format(*(inf2[0:2] + [channel]))

    corr_trace_name = "{}--{}.sac".format(sta1, sta2)
    corr_trace_name = os.path.join(source_conf['source_path'],
                                   'iteration_' + str(step), 'corr',
                                   corr_trace_name)
    return corr_trace_name


def get_ns(wf1, source_conf, insta=False):
    # Nr of time steps in traces
    if insta:
        # get path to instaseis db
        conf = yaml.safe_load(open(os.path.join(source_conf['project_path'],
                                                'config.yml')))
        dbpath = conf['wavefield_path']

        # open
        db = instaseis.open_db(dbpath)
        # get a test seismogram to determine...
        stest = db.get_seismograms(source=instaseis.ForceSource(latitude=0.0,
                                                                longitude=0.),
                                   receiver=instaseis.Receiver(latitude=10.,
                                                               longitude=0.),
                                   dt=1. / conf['wavefield_sampling_rate'])[0]

        nt = stest.stats.npts
        Fs = stest.stats.sampling_rate
    else:
        with WaveField(wf1) as wf1:
            nt = int(wf1.stats['nt'])
            Fs = round(wf1.stats['Fs'], 8)

    # Necessary length of zero padding
    # for carrying out frequency domain correlations/convolutions
    n = next_fast_len(2 * nt - 1)

    # Number of time steps for synthetic correlation
    n_lag = int(source_conf['max_lag'] * Fs)
    if nt - 2 * n_lag <= 0:
        n_lag = nt // 2
        warn('Resetting maximum lag to %g seconds:\
 Synthetics are too short for %g seconds.' % (n_lag / Fs, n_lag / Fs))

    n_corr = 2 * n_lag + 1

    return nt, n, n_corr, Fs


def g1g2_corr(wf1, wf2, corr_file, src, source_conf, insta=False):
    """
    Compute noise cross-correlations from two .h5 'wavefield' files.
    Noise source distribution and spectrum is given by starting_model.h5
    It is assumed that noise sources are delta-correlated in space.

    Metainformation: Include the reference station names for both stations
    from wavefield files, if possible. Do not include geographic information
    from .csv file as this might be error-prone. Just add the geographic
    info later if needed.
    """
    with open(os.path.join(source_conf['project_path'],
                           'config.yml')) as fh:
        conf = yaml.safe_load(fh)

    with NoiseSource(src) as nsrc:

        ntime, n, n_corr, Fs = get_ns(wf1, source_conf, insta)

        # use a one-sided taper: The seismogram probably has a non-zero end,
        # being cut off wherever the solver stopped running.
        taper = cosine_taper(ntime, p=0.01)
        taper[0: ntime // 2] = 1.0
        ntraces = nsrc.src_loc[0].shape[0]
        correlation = np.zeros(n_corr)

        if insta:
            # open database
            dbpath = conf['wavefield_path']

            # open
            db = instaseis.open_db(dbpath)
            # get receiver locations
            lat1 = geograph_to_geocent(float(wf1[2]))
            lon1 = float(wf1[3])
            rec1 = instaseis.Receiver(latitude=lat1, longitude=lon1)
            lat2 = geograph_to_geocent(float(wf2[2]))
            lon2 = float(wf2[3])
            rec2 = instaseis.Receiver(latitude=lat2, longitude=lon2)

        else:
            wf1 = WaveField(wf1)
            wf2 = WaveField(wf2)
            # Make sure all is consistent
            if False in (wf1.sourcegrid[1, 0:10] == wf2.sourcegrid[1, 0:10]):
                raise ValueError("Wave fields not consistent.")

            if False in (wf1.sourcegrid[1, -10:] == wf2.sourcegrid[1, -10:]):
                raise ValueError("Wave fields not consistent.")

            if False in (wf1.sourcegrid[0, -10:] == nsrc.src_loc[0, -10:]):
                raise ValueError("Wave field and source not consistent.")

        # Loop over source locations
        print_each_n = round(max(ntraces // 5, 1), -1)
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
                Fs = conf['wavefield_sampling_rate']
                s1 = db.get_seismograms(source=fsrc, receiver=rec1,
                                        dt=1. / Fs)[0].data * taper
                s2 = db.get_seismograms(source=fsrc, receiver=rec2,
                                        dt=1. / Fs)[0].data * taper
                s1 = np.ascontiguousarray(s1)
                s2 = np.ascontiguousarray(s2)

            else:
                # read Green's functions
                s1 = np.ascontiguousarray(wf1.data[i, :] * taper)
                s2 = np.ascontiguousarray(wf2.data[i, :] * taper)

            # Fourier transform for greater ease of convolution
            spec1 = np.fft.rfft(s1, n)
            spec2 = np.fft.rfft(s2, n)

            # convolve G1G2
            g1g2_tr = np.multiply(np.conjugate(spec1), spec2)

            # convolve noise source
            c = np.multiply(g1g2_tr, S)

            # transform back
            correlation += my_centered(np.fft.ifftshift(np.fft.irfft(c, n)),
                                       n_corr) * nsrc.surf_area[i]
            # occasional info
            if i % print_each_n == 0 and conf['verbose']:
                print("Finished {} of {} source locations.".format(i, ntraces))
# end of loop over all source locations #######################################

        if not insta:
            wf1.file.close()
            wf2.file.close()

        # save output
        trace = Trace()
        trace.stats.sampling_rate = Fs
        trace.data = correlation
        # try to add some meta data
        try:
            sta1 = wf1.stats['reference_station']
            sta2 = wf2.stats['reference_station']
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

        trace.write(filename=corr_file, format='SAC')


def run_corr(args):

    source_configfile = os.path.join(args.source_model, 'source_config.yml')
    step = int(args.step)
    steplengthrun = args.steplengthrun
    ignore_network = args.ignore_network

    # simple embarrassingly parallel run:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # get configuration
    source_config = yaml.safe_load(open(source_configfile))
    config = yaml.safe_load(open(os.path.join(source_config['project_path'],
                                              'config.yml')))
    obs_only = source_config['model_observed_only']
    channel = 'MX' + config['wavefield_channel']
    auto_corr = False  # default value
    try:
        auto_corr = source_config['get_auto_corr']
    except KeyError:
        pass

    # get possible station pairs
    p = define_correlationpairs(source_config['project_path'],
                                auto_corr=auto_corr)

    if rank == 0:
        print('Nr all possible correlation pairs %g ' % len(p))

    # Remove pairs for which no observation is available
    if obs_only:
        if steplengthrun:
            directory = os.path.join(source_config['source_path'],
                                     'observed_correlations_slt')
        else:
            directory = os.path.join(source_config['source_path'],
                                     'observed_correlations')
        if rank == 0:
            # split p into size lists for comm.scatter()
            p_split = np.array_split(p, size)
            p_split = [k.tolist() for k in p_split]
        else:
            p_split = None

        # scatter p_split to ranks
        p_split = comm.scatter(p_split, root=0)

        p_split = rem_no_obs(p_split, source_config, directory=directory)

        # gather all on rank 0
        p_new = comm.gather(list(p_split), root=0)

        # put all back into one array p
        if rank == 0:
            p = [i for j in p_new for i in j]

        # broadcast p to all ranks
        p = comm.bcast(p, root=0)
        if rank == 0:
            print('Nr correlation pairs after checking available observ. %g '
                  % len(p))

    # Remove pairs that have already been calculated
    p = rem_fin_prs(p, source_config, step)
    if rank == 0:
        print('Nr correlation pairs after checking already calculated ones %g'
              % len(p))
        print(16 * '*')

    if len(p) == 0:
        return()
    # The assignment of station pairs should be such that one core has as
    # many occurrences of the same station as possible;
    # this will prevent that many processes try to read from the same hdf5
    # file all at once.
    num_pairs = int(ceil(float(len(p)) / float(size)))

    p_p = p[rank * num_pairs: rank * num_pairs + num_pairs]

    if config['verbose']:
        print('Rank number %g' % rank)
        print('working on pair nr. %g to %g of %g.' % (rank * num_pairs,
                                                       rank * num_pairs +
                                                       num_pairs, len(p)))

    for cp in p_p:
        # try except is used here because of the massively parallel loop.
        # it needs to tolerate a couple of messups (e.g. a wavefield is
        # requested that isn't in the database)
        # if unknown errors occur and no correlations are computed, comment
        # out try-except to see the error messages.
        try:
            wf1, wf2, src = paths_input(cp, source_config,
                                        step, ignore_network)

            corr = path_output(cp, source_config, step, channel)
            print(corr)
        except (IndexError, FileNotFoundError):
            warn('Could not determine correlation for: %s\
\nCheck if wavefield .h5 file is available.' % cp)
            continue

        if os.path.exists(corr):
            continue

        g1g2_corr(wf1, wf2, corr, src, source_config)
    return()
