from __future__ import print_function
# from mpi4py import MPI
import numpy as np
import os
from glob import glob
from math import ceil

from obspy import read, Stream
from noisi_v1 import NoiseSource, WaveField
from obspy.signal.invsim import cosine_taper
# from warnings import warn
from noisi_v1.util.windows import my_centered
from noisi_v1.util.geo import geograph_to_geocent
from noisi_v1.util.corr_pairs import define_correlationpairs, rem_no_obs
from noisi_v1.scripts.correlation import config_params, get_ns
try:
    import instaseis
except ImportError:
    pass

# prepare embarrassingly parallel run:
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()


def add_input_files(kp, all_conf, insta=False):

    inf1 = kp[0].split()
    inf2 = kp[1].split()

    channel = 'MX' + all_conf.config['wavefield_channel']

    # station names
    if all_conf.ignore_network:
        sta1 = "*.{}..{}".format(*(inf1[1:2] + [channel]))
        sta2 = "*.{}..{}".format(*(inf2[1:2] + [channel]))
    else:
        sta1 = "{}.{}..{}".format(*(inf1[0:2] + [channel]))
        sta2 = "{}.{}..{}".format(*(inf2[0:2] + [channel]))

    # Wavefield files
    if not insta:
        dir = os.path.join(all_conf.config['project_path'], 'greens')
        wf1 = glob(os.path.join(dir, sta1 + '.h5'))[0]
        wf2 = glob(os.path.join(dir, sta2 + '.h5'))[0]
    else:
        # need to return two receiver coordinate pairs.
        # For buried sensors, depth could be used but no elevation is possible,
        # so maybe keep everything at 0 m?
        # lists of information directly from the stations.txt file.
        wf1 = inf1
        wf2 = inf2

    iteration_dir = os.path.join(all_conf.source_config['source_path'],
                                 'iteration_' + str(all_conf.step))
    # Adjoint source
    if all_conf.measr_config['mtype'] in ['energy_diff']:
        adj_src_basicnames = [os.path.join(iteration_dir, 'adjt',
                                           "{}--{}.c".format(sta1, sta2)),
                              os.path.join(iteration_dir, 'adjt',
                                           "{}--{}.a".format(sta1, sta2))]
    else:
        adj_src_basicnames = [os.path.join(iteration_dir, 'adjt',
                                           "{}--{}".format(sta1, sta2))]

    return(wf1, wf2, adj_src_basicnames)


def add_output_file(kp, all_conf, insta=False):

    id1 = kp[0].split()[0] + kp[0].split()[1]
    id2 = kp[1].split()[0] + kp[1].split()[1]

    if id1 < id2:
        inf1 = kp[0].split()
        inf2 = kp[1].split()
    else:
        inf2 = kp[0].split()
        inf1 = kp[1].split()

    channel = 'MX' + all_conf.config['wavefield_channel']
    sta1 = "{}.{}..{}".format(*(inf1[0:2] + [channel]))
    sta2 = "{}.{}..{}".format(*(inf2[0:2] + [channel]))

    kern_basicname = "{}--{}".format(sta1, sta2)
    kern_basicname = os.path.join(all_conf.source_config['source_path'],
                                  'iteration_' + str(all_conf.step), 'kern',
                                  kern_basicname)
    return (kern_basicname)


def compute_kernel(input_files, all_conf, nsrc, all_ns, taper,
                   insta=False):

    ntime, n, n_corr, Fs = all_ns
    wf1, wf2, adjt = input_files
########################################################################
# Prepare filenames and adjoint sources
########################################################################

    adjt_srcs = []

    for ix_f in range(all_conf.filtcnt):
        f = Stream()
        for a in adjt:
            adjtfile = a + '*.{}.sac'.format(ix_f)
            adjtfile = glob(adjtfile)
            try:
                f += read(adjtfile[0])[0]
                f[-1].data = my_centered(f[-1].data, n_corr)
            except IndexError:
                if all_conf.config['verbose']:
                    print('No adjoint source found: {}\n'.format(a))
                else:
                    pass
        if len(f) > 0:
            adjt_srcs.append(f)
        else:
            return None

    # Uniform spatial weights. (current model is in the adjoint source)
    nsrc.distr_basis = np.ones(nsrc.distr_basis.shape)
    ntraces = nsrc.src_loc[0].shape[0]

    if insta:
        # open database
        dbpath = all_conf.config['wavefield_path']
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
        wf1 = WaveField(wf1)
        wf2 = WaveField(wf2)
        # Make sure all is consistent
        if False in (wf1.sourcegrid[1, 0:10] == wf2.sourcegrid[1, 0:10]):
            raise ValueError("Wave fields not consistent.")

        if False in (wf1.sourcegrid[1, -10:] == wf2.sourcegrid[1, -10:]):
            raise ValueError("Wave fields not consistent.")

        if False in (wf1.sourcegrid[0, -10:] == nsrc.src_loc[0, -10:]):
            raise ValueError("Wave field and source not consistent.")

    kern = np.zeros((all_conf.filtcnt, ntraces, len(adjt)))

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
            dt = 1. / all_conf.source_config['sampling_rate']
            s1 = db.get_seismograms(source=fsrc, receiver=rec1,
                                    dt=dt)[0].data * taper
            s1 = np.ascontiguousarray(s1)
            s2 = db.get_seismograms(source=fsrc, receiver=rec2,
                                    dt=dt)[0].data * taper
            s2 = np.ascontiguousarray(s2)
            spec1 = np.fft.rfft(s1, n)
            spec2 = np.fft.rfft(s2, n)

        else:
            if not wf1.fdomain:
                s1 = np.ascontiguousarray(wf1.data[i, :] * taper)
                s2 = np.ascontiguousarray(wf2.data[i, :] * taper)
                spec1 = np.fft.rfft(s1, n)
                spec2 = np.fft.rfft(s2, n)
            else:
                spec1 = wf1.data[i, :]
                spec2 = wf2.data[i, :]

        g1g2_tr = np.multiply(np.conjugate(spec1), spec2)
        # spectrum -- to be replaced by spectral basis function
        c = np.multiply(g1g2_tr, S)
    #######################################################################
    # Get Kernel at that location
    #######################################################################
        corr_temp = my_centered(np.fft.ifftshift(np.fft.irfft(c, n)),
                                n_corr) * nsrc.surf_area[i]

    #######################################################################
    # Apply the 'adjoint source'
    #######################################################################
        for ix_f in range(all_conf.filtcnt):
            f = adjt_srcs[ix_f]

            if f is None:
                continue

            for j in range(len(f)):
                delta = f[j].stats.delta
                kern[ix_f, i, j] = np.dot(corr_temp, f[j].data) * delta

        if i % print_each_n == 0 and all_conf.config['verbose']:
            print("Finished {} of {} source locations.".format(i, ntraces))
    if not insta:
        wf1.file.close()
        wf2.file.close()

    return kern


def define_kernel_tasks(all_conf, comm, size, rank):

    p = define_correlationpairs(all_conf.source_config['project_path'],
                                auto_corr=all_conf.auto_corr)
    if rank == 0 and all_conf.config['verbose']:
        print('Nr all possible kernels %g ' % len(p))

    # Remove pairs for which no observation is available
    if all_conf.source_config['model_observed_only']:

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
            print('Nr kernels after checking available observ. %g ' % len(p))

    # The assignment of station pairs should be such that one core
    # has as many occurrences of the same station as possible;
    # this will prevent that many processes try to access the
    # same hdf5 file all at once.
    num_pairs = int(ceil(float(len(p)) / float(size)))
    p_p = p[rank * num_pairs: rank * num_pairs + num_pairs]

    return p_p, num_pairs, len(p)


def run_kern(args, comm, size, rank):

    args.steplengthrun = False  # by default
    all_conf = config_params(args, comm, size, rank)

    kernel_tasks, n_p, n_p_p = define_kernel_tasks(all_conf, comm, size, rank)
    if len(kernel_tasks) == 0:
        return()
    if all_conf.config['verbose']:
        print('Rank number %g' % rank)
        print('working on pair nr. %g to %g of %g.' % (rank * n_p_p,
                                                       rank * n_p_p +
                                                       n_p_p, n_p))

    # Current model for the noise source
    nsrc = os.path.join(all_conf.source_config['project_path'],
                        all_conf.source_config['source_name'],
                        'spectral_model.h5')

    # Smart numbers
    all_ns = get_ns(all_conf)  # ntime, n, n_corr, Fs

    # use a one-sided taper: The seismogram probably has a non-zero end,
    # being cut off wherever the solver stopped running.
    taper = cosine_taper(all_ns[0], p=0.01)
    taper[0: all_ns[0] // 2] = 1.0

    with NoiseSource(nsrc) as nsrc:
        for kp in kernel_tasks:
            try:
                input_files = add_input_files(kp, all_conf)
                output_file = add_output_file(kp, all_conf)
            except (IOError, IndexError):
                if all_conf.config['verbose']:
                    print('Could not find input for: %s\
\nCheck if wavefield .h5 file and base_model file are available.' % kp)
                continue

            kern = compute_kernel(input_files, all_conf, nsrc, all_ns, taper)
            if kern is None:
                continue

            for ix_f in range(all_conf.filtcnt):
                if kern[ix_f, :, :].sum() != 0:
                    filename = output_file + '.{}.npy'.format(ix_f)
                    np.save(filename, kern[ix_f, :, :])
    return()
