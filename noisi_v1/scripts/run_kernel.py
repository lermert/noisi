from __future__ import print_function
from mpi4py import MPI
import numpy as np
import os
import yaml
from glob import glob
from math import ceil

from obspy import read, Stream
from noisi_v1 import NoiseSource, WaveField
from obspy.signal.invsim import cosine_taper
from warnings import warn
from noisi_v1.scripts.run_correlation import get_ns
from noisi_v1.util.windows import my_centered
from noisi_v1.util.geo import geograph_to_geocent
from noisi_v1.util.corr_pairs import *
try:
    import instaseis
except ImportError:
    pass


def paths_input(cp, source_conf, step, ignore_network, instaseis=False):

    inf1 = cp[0].split()
    inf2 = cp[1].split()

    conf = yaml.safe_load(open(os.path.join(source_conf['project_path'],
                                            'config.yml')))
    measr_conf = yaml.safe_load(open(os.path.join(source_conf['source_path'],
                                                  'measr_config.yml')))

    channel = 'MX' + conf['wavefield_channel']

    # station names
    if ignore_network:
        sta1 = "*.{}..{}".format(*(inf1[1:2] + [channel]))
        sta2 = "*.{}..{}".format(*(inf2[1:2] + [channel]))
    else:
        sta1 = "{}.{}..{}".format(*(inf1[0:2] + [channel]))
        sta2 = "{}.{}..{}".format(*(inf2[0:2] + [channel]))

    # Wavefield files
    if not instaseis:
        dir = os.path.join(conf['project_path'], 'greens')
        wf1 = glob(os.path.join(dir, sta1 + '.h5'))[0]
        wf2 = glob(os.path.join(dir, sta2 + '.h5'))[0]
    else:
        # need to return two receiver coordinate pairs.
        # For buried sensors, depth could be used but no elevation is possible,
        # so maybe keep everything at 0 m?
        # lists of information directly from the stations.txt file.
        wf1 = inf1
        wf2 = inf2

    nsrc = os.path.join(source_conf['project_path'],
                        source_conf['source_name'], 'iteration_' + str(step),
                        'starting_model.h5')

    # Adjoint source
    if measr_conf['mtype'] in ['energy_diff', 'envelope']:
        adj_src_basicnames = [os.path.join(source_conf['source_path'],
                                           'iteration_' + str(step),
                                           'adjt',
                                           "{}--{}.c".format(sta1, sta2)),
                              os.path.join(source_conf['source_path'],
                                           'iteration_' + str(step),
                                           'adjt',
                                           "{}--{}.a".format(sta1, sta2))]
    else:
        adj_src_basicnames = [os.path.join(source_conf['source_path'],
                                           'iteration_' + str(step),
                                           'adjt',
                                           "{}--{}".format(sta1, sta2))]

    return(wf1, wf2, nsrc, adj_src_basicnames)


def paths_output(cp, source_conf, step):

    conf = yaml.safe_load(open(os.path.join(source_conf['project_path'],
                                            'config.yml')))
    id1 = cp[0].split()[0] + cp[0].split()[1]
    id2 = cp[1].split()[0] + cp[1].split()[1]

    if id1 < id2:
        inf1 = cp[0].split()
        inf2 = cp[1].split()
    else:
        inf2 = cp[0].split()
        inf1 = cp[1].split()

    channel = 'MX' + conf['wavefield_channel']
    sta1 = "{}.{}..{}".format(*(inf1[0:2] + [channel]))
    sta2 = "{}.{}..{}".format(*(inf2[0:2] + [channel]))

    kern_basicname = "{}--{}".format(sta1, sta2)
    kern_basicname = os.path.join(source_conf['source_path'],
                                  'iteration_' + str(step), 'kern',
                                  kern_basicname)
    return (kern_basicname)


def g1g2_kern(wf1str, wf2str, kernel, adjt, src, source_conf, insta=False):

    measr_conf = yaml.safe_load(open(os.path.join(source_conf['source_path'],
                                                  'measr_config.yml')))
    bandpass = measr_conf['bandpass']

    conf = yaml.safe_load(open(os.path.join(source_conf['project_path'],
                                            'config.yml')))

    if bandpass is None:
        filtcnt = 1
    elif type(bandpass) == list:
        if type(bandpass[0]) != list:
            filtcnt = 1
        else:
            filtcnt = len(bandpass)

    ntime, n, n_corr, Fs = get_ns(wf1str, source_conf, insta)
    # use a one-sided taper: The seismogram probably has a non-zero end,
    # being cut off whereever the solver stopped running.
    taper = cosine_taper(ntime, p=0.01)
    taper[0:ntime // 2] = 1.0

########################################################################
# Prepare filenames and adjoint sources
########################################################################

    filenames = []
    adjt_srcs = []

    for ix_f in range(filtcnt):
        filename = kernel + '.{}.npy'.format(ix_f)
        filenames.append(filename)

        f = Stream()
        for a in adjt:
            adjtfile = a + '*.{}.sac'.format(ix_f)
            adjtfile = glob(adjtfile)
            try:
                f += read(adjtfile[0])[0]
                f[-1].data = my_centered(f[-1].data, n_corr)
            except IndexError:
                warn('No adjoint source found: {}\n'.format(a))

        if len(f) > 0:
            adjt_srcs.append(f)
        else:
            return()


########################################################################
# Compute the kernels
########################################################################

    with NoiseSource(src) as nsrc:
        # Uniform spatial weights.
        nsrc.distr_basis = np.ones(nsrc.distr_basis.shape)
        ntraces = nsrc.src_loc[0].shape[0]

        if insta:
            # open database
            dbpath = conf['wavefield_path']
            # open and determine Fs, nt
            db = instaseis.open_db(dbpath)
            # get receiver locations
            lat1 = geograph_to_geocent(float(wf1[2]))
            lon1 = float(wf1[3])
            rec1 = instaseis.Receiver(latitude=lat1, longitude=lon1)
            lat2 = geograph_to_geocent(float(wf2[2]))
            lon2 = float(wf2[3])
            rec2 = instaseis.Receiver(latitude=lat2, longitude=lon2)

        else:
            wf1 = WaveField(wf1str)
            wf2 = WaveField(wf2str)
            # Make sure all is consistent
            if False in (wf1.sourcegrid[1, 0:10] == wf2.sourcegrid[1, 0:10]):
                raise ValueError("Wave fields not consistent.")

            if False in (wf1.sourcegrid[1, -10:] == wf2.sourcegrid[1, -10:]):
                raise ValueError("Wave fields not consistent.")

            if False in (wf1.sourcegrid[0, -10:] == nsrc.src_loc[0, -10:]):
                raise ValueError("Wave field and source not consistent.")

        kern = np.zeros((filtcnt, ntraces, len(adjt)))

        # Loop over locations
        print_each_n = max(5, round(max(ntraces // 5, 1), -1))
        for i in range(ntraces):

            # noise source spectrum at this location
            # For the kernel, this contains only the basis functions of the
            # spectrum without weights; might still be location-dependent,
            # for example when constraining sensivity to ocean
            S = nsrc.get_spect(i)

            if S.sum() == 0.:
                # The spectrum has 0 phase so only checking
                # absolute value here
                continue

            if insta:
                # get source locations
                lat_src = geograph_to_geocent(nsrc.src_loc[1, i])
                lon_src = nsrc.src_loc[0, i]
                fsrc = instaseis.ForceSource(latitude=lat_src,
                                             longitude=lon_src, f_r=1.e12)
                dt = 1. / source_conf['sampling_rate']
                s1 = db.get_seismograms(source=fsrc, receiver=rec1,
                                        dt=dt)[0].data * taper
                s1 = np.ascontiguousarray(s1)
                s2 = db.get_seismograms(source=fsrc, receiver=rec2,
                                        dt=dt)[0].data * taper
                s2 = np.ascontiguousarray(s2)

            else:
                s1 = np.ascontiguousarray(wf1.data[i, :] * taper)
                s2 = np.ascontiguousarray(wf2.data[i, :] * taper)

            spec1 = np.fft.rfft(s1, n)
            spec2 = np.fft.rfft(s2, n)

            g1g2_tr = np.multiply(np.conjugate(spec1), spec2)
            c = np.multiply(g1g2_tr, S)

        #######################################################################
        # Get Kernel at that location
        #######################################################################
            corr_temp = my_centered(np.fft.ifftshift(np.fft.irfft(c, n)),
                                    n_corr)

        #######################################################################
        # Apply the 'adjoint source'
        #######################################################################
            for ix_f in range(filtcnt):
                f = adjt_srcs[ix_f]

                if f is None:
                    continue

                for j in range(len(f)):
                    delta = f[j].stats.delta
                    kern[ix_f, i, j] = np.dot(corr_temp, f[j].data) * delta

            if i % print_each_n == 0 and conf['verbose']:
                print("Finished {} of {} source locations.".format(i, ntraces))

    if not insta:
        wf1.file.close()
        wf2.file.close()

    for ix_f in range(filtcnt):
        filename = filenames[ix_f]
        if kern[ix_f, :, :].sum() != 0:
            np.save(filename, kern[ix_f, :, :])
    return()


def run_kern(args):

    # simple embarrassingly parallel run:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    source_configfile = os.path.join(args.source_model, "source_config.yml")
    step = int(args.step)
    ignore_network = args.ignore_network

    source_config = yaml.safe_load(open(source_configfile))
    configfile = os.path.join(source_config['project_path'],
                              'config.yml')
    config = yaml.safe_load(open(configfile))

    obs_only = source_config['model_observed_only']
    auto_corr = False
    try:
        auto_corr = source_config['get_auto_corr']
    except KeyError:
        pass

    p = define_correlationpairs(source_config['project_path'],
                                auto_corr=auto_corr)
    if rank == 0:
        print('Nr all possible kernels %g ' % len(p))

    # Remove pairs for which no observation is available
    if obs_only:
        if rank == 0:
            # split p into size lists for comm.scatter()
            p_split = np.array_split(p, size)
            p_split = [k.tolist() for k in p_split]
        else:
            p_split = None

        # scatter p_split to ranks
        p_split = comm.scatter(p_split, root=0)
        directory = os.path.join(source_config['source_path'],
                                 'observed_correlations')
        p_split = rem_no_obs(p_split, source_config, directory=directory)

        # gather all on rank 0
        p_new = comm.gather(list(p_split), root=0)

        # put all back into one array p
        if rank == 0:
            p = [i for j in p_new for i in j]

        # broadcast p to all ranks
        p = comm.bcast(p, root=0)
        if rank == 0:
            print('Nr kernels after checking available observ. %g ' % len(p))

    # The assignment of station pairs should be such that one core
    # has as many occurrences of the same station as possible;
    # this will prevent that many processes try to access the
    # same hdf5 file all at once.
    num_pairs = int(ceil(float(len(p)) / float(size)))
    p_p = p[rank * num_pairs: rank * num_pairs + num_pairs]

    if config['verbose']:
        print('Rank number %g' % rank)
        print('working on pair nr. %g to %g of %g.' % (rank * num_pairs,
                                                       rank * num_pairs +
                                                       num_pairs, len(p)))

    for cp in p_p:
        try:
            wf1, wf2, src, adjt = paths_input(cp, source_config,
                                              step, ignore_network)
            kernel = paths_output(cp, source_config, step)

        except (IOError, IndexError):
            warn('Could not find input for: %s\
#\nCheck if wavefield .h5 file and base_model file are available.' % cp)
#            continue
        g1g2_kern(wf1, wf2, kernel, adjt, src, source_config)

    return()
