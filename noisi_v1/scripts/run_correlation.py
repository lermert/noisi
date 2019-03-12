from mpi4py import MPI
import numpy as np
import os
import json
from glob import glob
from math import ceil
from scipy.fftpack import next_fast_len
from obspy import Trace
from noisi_v1 import NoiseSource, WaveField
from obspy.signal.invsim import cosine_taper
from noisi_v1.util.windows import my_centered
from noisi_v1.util.geo import geograph_to_geocent
from noisi_v1.util.corr_pairs import define_correlationpairs
from noisi_v1.util.corr_pairs import rem_fin_prs, rem_no_obs
import instaseis


def paths_input(cp, source_conf, step, ignore_network, instaseis):

    inf1 = cp[0].split()
    inf2 = cp[1].split()

    conf = json.load(open(os.path.join(source_conf['project_path'],
                                       'config.json')))
    channel = source_conf['channel']

    # station names
    if ignore_network:
        sta1 = "*.{}..{}".format(*(inf1[1:2] + [channel]))
        sta2 = "*.{}..{}".format(*(inf2[1:2] + [channel]))
    else:
        sta1 = "{}.{}..{}".format(*(inf1[0:2] + [channel]))
        sta2 = "{}.{}..{}".format(*(inf2[0:2] + [channel]))

    # Wavefield files
    if not instaseis:
        if source_conf['preprocess_do']:
            dir = os.path.join(source_conf['source_path'],
                               'wavefield_processed')
        else:
            dir = conf['wavefield_path']

        wf1 = glob(os.path.join(dir, sta1 + '.h5'))[0]
        wf2 = glob(os.path.join(dir, sta2 + '.h5'))[0]
    else:
        # need to return two receiver coordinate pairs. For buried sensors, depth could be used but no elevation is possible.
        # so maybe keep everything at 0 m?
        # lists of information directly from the stations.txt file.
        wf1 = inf1
        wf2 = inf2

    # Starting model for the noise source
    nsrc = os.path.join(source_conf['project_path'],
                        source_conf['source_name'], 'step_' + str(step),
                        'starting_model.h5')
    return(wf1, wf2, nsrc)


def path_output(cp, source_conf, step):

    id1 = cp[0].split()[0] + cp[0].split()[1]
    id2 = cp[1].split()[0] + cp[1].split()[1]

    if id1 < id2:
        inf1 = cp[0].split()
        inf2 = cp[1].split()
    else:
        inf2 = cp[0].split()
        inf1 = cp[1].split()

    channel = source_conf['channel']
    sta1 = "{}.{}..{}".format(*(inf1[0:2] + [channel]))
    sta2 = "{}.{}..{}".format(*(inf2[0:2] + [channel]))


    corr_trace_name = "{}--{}.sac".format(sta1, sta2)
    corr_trace_name = os.path.join(source_conf['source_path'],
                                    'step_' + str(step), 'corr',
                                    corr_trace_name)
    return corr_trace_name


def get_ns(wf1, source_conf, insta):
    # Nr of time steps in traces
    if insta:
        # get path to instaseis db
        # ToDo: ugly.
        dbpath = json.load(open(
                                os.path.join(source_conf['project_path'],
                                'config.json')))['wavefield_path']
        # open 
        db = instaseis.open_db(dbpath)
        # get a test seismogram to determine...
        stest = db.get_seismograms(source=instaseis.ForceSource(latitude=0.0,
                                                                longitude=0.),
                                   receiver=instaseis.Receiver(latitude=10.,
                                                               longitude=0.),
                                   dt=1. / source_conf['sampling_rate'])[0]

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
        print('Resetting maximum lag to %g seconds:\
 Synthetics are too short for a maximum lag of %g seconds.'
              % (nt // 2 / Fs, n_lag / Fs)) 
    n_lag = nt // 2

    n_corr = 2 * n_lag + 1

    return nt,n,n_corr,Fs
        
    
def g1g2_corr(wf1,wf2,corr_file,src,source_conf,insta):
    """
    Compute noise cross-correlations from two .h5 'wavefield' files.
    Noise source distribution and spectrum is given by starting_model.h5
    It is assumed that noise sources are delta-correlated in space.

    Metainformation: Include the reference station names for both stations
    from wavefield files, if possible. Do not include geographic information
    from .csv file as this might be error-prone. Just add the geographic
    info later if needed.
    """

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
            dbpath = json.load(open(
                     os.path.join(source_conf['project_path'],
                                 'config.json')))['wavefield_path']
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

        # Loop over source locations
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

                s1 = np.ascontiguousarray(db.get_seismograms(source=fsrc,
                      receiver=rec1, 
                      dt=1. / source_conf['sampling_rate'])[0].data * taper)
                s2 = np.ascontiguousarray(db.get_seismograms(source=fsrc,
                    receiver=rec2,
                    dt=1. / source_conf['sampling_rate'])[0].data * taper)

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
            if i % 50000 == 0:
                print("Finished {} source locations.".format(i))
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
        except KeyError:
            pass

        trace.write(filename=corr_file, format='SAC')


def run_corr(args):

    source_configfile = os.path.join(args.source_model, 'source_config.json')
    step = int(args.step)
    steplengthrun = args.steplengthrun
    ignore_network = args.ignore_network

    # simple embarrassingly parallel run:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # get configuration
    source_config = json.load(open(source_configfile))
    obs_only = source_config['model_observed_only']
    insta = json.load(open(os.path.join(source_config['project_path'],
        'config.json')))['instaseis']
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
    if obs_only and not steplengthrun:
        directory = os.path.join(source_config['source_path'],
                                 'observed_correlations')

        p = rem_no_obs(p, source_config, directory=directory)
        if rank == 0:
            print('Nr correlation pairs after checking\
 available observ. %g ' % len(p))

    if steplengthrun:
        directory = os.path.join(source_config['source_path'],
                                 'step_' + str(step), 'obs_slt')

        p = rem_no_obs(p, source_config, directory=directory)
        if rank == 0:
            print('Nr correlation pairs after checking\
 available observ. %g ' % len(p))

    # Remove pairs that have already been calculated
    p = rem_fin_prs(p, source_config, step)
    if rank == 0:
        print('Nr correlation pairs after checking already calculated ones %g'
              % len(p))
        print(16 * '*')

    # The assignment of station pairs should be such that one core has as
    # many occurrences of the same station as possible;
    # this will prevent that many processes try to read from the same hdf5
    # file all at once.
    num_pairs = int(ceil(float(len(p)) / float(size)))

    p_p = p[rank * num_pairs: rank * num_pairs + num_pairs]

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
                                        step, ignore_network, insta)

            corr = path_output(cp, source_config, step)
            print(corr)

        except:
            print('Could not determine correlation for: %s\
\nCheck if wavefield .h5 file is available.' % cp)
            continue

        if os.path.exists(corr):
            continue

        g1g2_corr(wf1, wf2, corr, src, source_config, insta=insta)
    return()
