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


def measurement(source_config, mtype, step, ignore_net,
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
        msr_o = func(tr_o, **options)
        msr_s = func(tr_s, **options)

        # Get the adjoint source
        adjt_func = am.get_adj_func(mtype)
        adjt, success = adjt_func(tr_o, tr_s, **options)
        if not success:
            continue

        # timeseries-like measurements:
        if mtype in ['square_envelope',
                     'full_waveform', 'windowed_waveform']:
            l2_so = 0.5 * np.sum(np.power((msr_s - msr_o), 2))
            snr = snratio(tr_o, **options)
            snr_a = snratio(tr_o, **_options_ac)
            info.extend([np.nan, np.nan, np.nan, np.nan,
                         l2_so, snr, snr_a, tr_o.stats.sac.user0])
            adjoint_source[0].data = adjt

        # single value measurements:
        else:
            if mtype == 'energy_diff':
                l2_so = 0.5 * (msr_s - msr_o)**2 / (msr_o) ** 2
                msr = msr_o[0]
                msr_a = msr_o[1]
                snr = snratio(tr_o, **options)
                snr_a = snratio(tr_o, **_options_ac)
                l2 = l2_so.sum()
                info.extend([msr_s[0], msr_s[1], msr, msr_a,
                             l2, snr, snr_a, tr_o.stats.sac.user0])

                adjoint_source += adjoint_source[0].copy()
                for ix_branch in range(2):
                    adjoint_source[ix_branch].data = adjt[ix_branch]
                    adjoint_source[ix_branch].data *= (msr_s[ix_branch] -
                                                       msr_o[ix_branch]) / msr_o[ix_branch] ** 2

            elif mtype == 'ln_energy_ratio':
                l2_so = 0.5 * (msr_s - msr_o)**2
                msr = msr_o
                snr = snratio(tr_o, **options)
                snr_a = snratio(tr_o, **_options_ac)
                info.extend([msr_s, np.nan, msr, np.nan,
                             l2_so, snr, snr_a, tr_o.stats.sac.user0])
                adjt *= (msr_s - msr_o)
                adjoint_source[0].data = adjt

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

    if measr_config['mtype'] in ['square_envelope']:
        window_params['win_overlap'] = True

    hws = window_params['hw'][:]

    for i in range(len(bandpass)):

        g_speed = g_speeds[i]
        window_params['hw'] = hws[i]
        ms = measurement(source_config, mtype, step, ignore_network,
                         ix_bandpass=i, bandpass=bandpass[i],
                         step_test=step_test, taper_perc=taper_perc,
                         g_speed=g_speed, window_params=window_params)

        filename = '{}.{}.measurement.csv'.format(mtype, i)
        ms.to_csv(os.path.join(step_dir, filename), index=None)

    return()
