import os
import numpy as np
import pandas as pd
import copy
import yaml
from obspy import read, Trace, Stream
from obspy.geodetics import gps2dist_azimuth
from noisi.scripts import measurements as rm
from noisi.scripts import adjnt_functs as am
from noisi.util.windows import my_centered, snratio
from noisi.util.corr_pairs import get_synthetics_filename
from warnings import warn


def get_station_info(stats):

    sta1 = '{}.{}.{}.{}'.format(stats.network, stats.station, stats.location,
                                stats.channel)

    try:
        sta2 = '{}.{}.{}.{}'.format(stats.sac.kuser0.strip(),
                                    stats.sac.kevnm.strip(),
                                    stats.sac.kuser1.strip(),
                                    stats.sac.kuser2.strip())
    except AttributeError:
        sta2 = '{}.{}.{}.{}'.format(stats.sac.kuser0.strip(),
                                    stats.sac.kevnm.strip(),
                                    '', stats.sac.kuser2.strip())
    lat1 = stats.sac.stla
    lon1 = stats.sac.stlo
    lat2 = stats.sac.evla
    lon2 = stats.sac.evlo
    dist = stats.sac.dist
    az, baz = gps2dist_azimuth(lat1, lon1, lat2, lon2)[1:]

    return([sta1, sta2, lat1, lon1, lat2, lon2, dist, az, baz])


def measurement(rank, size, source_config, mtype, step, ignore_net,
                ix_bandpass, bandpass, step_test, taper_perc, **options):

    """
    Get measurements on noise correlation data and synthetics.
    options: g_speed,window_params (only needed if
    mtype is ln_energy_ratio or enery_diff)
    """
    verbose = yaml.safe_load(open(os.path.join(source_config['project_path'],
                                               'config.yml')))['verbose']
    step_n = 'iteration_{}'.format(int(step))
    step_dir = os.path.join(source_config['source_path'], step_n)

    if step_test:
        corr_dir = os.path.join(step_dir, 'obs_slt')
    else:
        corr_dir = os.path.join(source_config['source_path'],
                                'observed_correlations')

    files = [f for f in os.listdir(corr_dir)]
    files = files[rank::size]
    files = [os.path.join(corr_dir, f) for f in files]
    synth_dir = os.path.join(step_dir, 'corr')

    columns = ['sta1', 'sta2', 'lat1', 'lon1', 'lat2', 'lon2', 'dist', 'az',
               'baz', 'syn', 'syn_a', 'obs', 'obs_a', 'l2_norm', 'snr',
               'snr_a', 'nstack']
    measurements = pd.DataFrame(columns=columns)

    options['window_params']['causal_side'] = True  # relic for signal to noise
    _options_ac = copy.deepcopy(options)
    _options_ac['window_params']['causal_side'] = (not(options['window_params']
                                                   ['causal_side']))

    if files == []:
        msg = 'No input found!'
        raise ValueError(msg)

    for i, f in enumerate(files):

        # Read data
        try:
            tr_o = read(f)[0]
        except IOError:
            if verbose:
                print('\nCould not read data: ' + os.path.basename(f))
            continue

        # Read synthetics
        synth_filename = get_synthetics_filename(os.path.basename(f),
                                                 synth_dir,
                                                 ignore_network=ignore_net)

        if synth_filename is None:
            continue

        try:
            tr_s = read(synth_filename)[0]
        except IOError:
            if verbose:
                print('\nCould not read synthetics: ' + synth_filename)
            continue

        # Assigning stats to synthetics, cutting them to right length
        tr_s.stats.sac = tr_o.stats.sac.copy()
        tr_s.data = my_centered(tr_s.data, tr_o.stats.npts)
        # Get all the necessary information
        info = get_station_info(tr_o.stats)
        # Collect the adjoint source
        adjoint_source = Stream()
        adjoint_source += Trace()
        adjoint_source[0].stats.sampling_rate = tr_s.stats.sampling_rate
        adjoint_source[0].stats.sac = tr_s.stats.sac.copy()

        # Filter
        if bandpass is not None:
            tr_o.taper(taper_perc / 100.)
            tr_o.filter('bandpass', freqmin=bandpass[0],
                        freqmax=bandpass[1], corners=bandpass[2],
                        zerophase=True)
            tr_s.taper(taper_perc / 100.)
            tr_s.filter('bandpass', freqmin=bandpass[0],
                        freqmax=bandpass[1], corners=bandpass[2],
                        zerophase=True)

        # Weight observed stack by nstack
        tr_o.data /= tr_o.stats.sac.user0

        # Take the measurement
        func = rm.get_measure_func(mtype)
        tr_o_m = tr_o.copy()
        tr_s_m = tr_s.copy()
        msr = func(tr_o_m, tr_s_m, **options)

        # Get the adjoint source
        adjt_func = am.get_adj_func(mtype)
        tr_o_m = tr_o.copy()
        tr_s_m = tr_s.copy()
        adjt, success = adjt_func(tr_o_m, tr_s_m, **options)
        if not success:
            continue

        if mtype in ["square_envelope", "full_waveform", "windowed_waveform",
                     "ln_energy_ratio", "envelope"]:
            snr = snratio(tr_o, **options)
            snr_a = snratio(tr_o, **_options_ac)
            info.extend([np.nan, np.nan, np.nan, np.nan,
                         msr, snr, snr_a, tr_o.stats.sac.user0])
            adjoint_source[0].data = adjt

        elif mtype == "energy_diff":
            snr = snratio(tr_o, **options)
            snr_a = snratio(tr_o, **_options_ac)
            info.extend([np.nan, np.nan, msr[0], msr[1],
                         msr.sum(), snr, snr_a, tr_o.stats.sac.user0])
            adjoint_source += adjoint_source[0].copy()
            for ix_branch in range(2):
                adjoint_source[ix_branch].data = adjt[ix_branch]
        measurements.loc[i] = info

        # save the adjoint source
        if len(adjoint_source) == 1:
            adjt_filename = os.path.basename(synth_filename).rstrip('sac') +\
                '{}.sac'.format(ix_bandpass)
            adjoint_source[0].write(os.path.join(step_dir, 'adjt',
                                                 adjt_filename), format='SAC')
        elif len(adjoint_source) == 2:
            for ix_branch, branch in enumerate(['c', 'a']):
                adjt_filename = os.path.basename(synth_filename).\
                    rstrip('sac') + '{}.{}.sac'.format(branch, ix_bandpass)
                adjoint_source[ix_branch].write(os.path.join(step_dir,
                                                             'adjt',
                                                             adjt_filename),
                                                format='SAC')
        else:
            raise ValueError("Some problem with adjoint sources.")

    return measurements


def run_measurement(args, comm, size, rank):

    source_configfile = os.path.join(args.source_model, "source_config.yml")
    step = int(args.step)
    step_test = args.steplengthrun
    ignore_network = args.ignore_network

    # get parameters
    source_config = yaml.safe_load(open(source_configfile))
    measr_configfile = os.path.join(source_config['source_path'],
                                    'measr_config.yml')
    measr_config = yaml.safe_load(open(measr_configfile))
    configfile = os.path.join(source_config['project_path'], 'config.yml')
    config = yaml.safe_load(open(configfile))
    mtype = measr_config['mtype']
    bandpass = measr_config['bandpass']
    step_n = 'iteration_{}'.format(int(step))
    step_dir = os.path.join(source_config['source_path'],
                            step_n)
    taper_perc = measr_config['taper_perc']

    window_params = {}
    window_params['hw'] = measr_config['window_params_hw']
    window_params['sep_noise'] = measr_config['window_params_sep_noise']
    window_params['win_overlap'] = measr_config['window_params_win_overlap']
    window_params['wtype'] = measr_config['window_params_wtype']
    window_params['plot'] = measr_config['window_plot_measurements']

    if bandpass is None:
        bandpass = [None]
    if type(bandpass[0]) != list and bandpass[0] is not None:
            bandpass = [bandpass]
            if config['verbose']:
                warn('\'Bandpass\' should be defined as list of filters.')

    if type(window_params['hw']) != list:
        window_params['hw'] = [window_params['hw']]

    if len(window_params['hw']) != len(bandpass):
        if config['verbose']:
            warn('Using the same window length for all measurements.')
        window_params['hw'] = len(bandpass) * [window_params['hw'][0]]

    if type(measr_config['g_speed']) in [float, int]:
        if config['verbose']:
            warn('Using the same group velocity for all measurements.')
        g_speeds = len(bandpass) * [measr_config['g_speed']]

    elif type(measr_config['g_speed']) == list\
        and len(measr_config['g_speed']) == len(bandpass):
        g_speeds = measr_config['g_speed']

    if measr_config['mtype'] in ['square_envelope', 'envelope']:
        window_params['win_overlap'] = True

    hws = window_params['hw'][:]

    for i in range(len(bandpass)):

        g_speed = g_speeds[i]
        window_params['hw'] = hws[i]
        ms = measurement(rank, size, source_config, mtype, step, ignore_network,
                         ix_bandpass=i, bandpass=bandpass[i],
                         step_test=step_test, taper_perc=taper_perc,
                         g_speed=g_speed, window_params=window_params)

        filename = 'temp.{}.measurement.csv'.format(rank)
        ms.to_csv(os.path.join(step_dir, filename), index=None)
        comm.barrier()
        dats = []
        if rank == 0:
            for ix_r in range(size):
                fname = os.path.join(step_dir, 'temp.{}.measurement.csv'.format(ix_r))
                dats.append(pd.read_csv(fname))
            dats = pd.concat(dats, ignore_index=True)
            filename = '{}.{}.measurement.csv'.format(mtype, i)
            dats.to_csv(os.path.join(step_dir, filename), index=None)
            os.system("rm " + os.path.join(step_dir, "temp.*.csv"))
    return()
