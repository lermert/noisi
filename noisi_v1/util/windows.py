import numpy as np
from warnings import warn


def my_centered(arr, newsize):

    # pad with zeros, if newsize > len(arr)
    newarr = np.zeros(newsize)

    # get the center portion of a 1-dimensional array correctly
    n = len(arr)
    i0 = (n - newsize) // 2
    if n % 2 == 0:
        # This is somehow a matter of definition,
        # because the array has no 'center' sample
        i0 += 1

    if i0 < 0:

        i0 = (newsize - n) // 2
        newarr[i0: i0 + n] += arr

    else:
        newarr[:] += arr[i0: i0 + newsize]
    return newarr


def window_checks(i0, i1, i2, i3, n, win_overlap):

    if n % 2 == 0:
        print('Correlation length must be 2*n+1, otherwise arbitrary middle \
sample. This correlation has length 2*n.')
        return(False)

    # Check if this will overlap with acausal side
    if i0 < n / 2 - 1 and win_overlap:
        msg = 'Windows of causal and acausal side overlap. Set win_overlap==False to \
skip these correlations.'
        warn(msg)
    elif i0 < n / 2 - 1 and not win_overlap:
        print('Windows overlap, skipping...')
        return(False)

    # Out of bounds?
    if i0 < 0 or i1 > n:
        print('\nNo windows found: Time series is too short.')
        return(False)
    elif i2 < 0 or i3 > n:
        print('\nNo windows found: Noise window not covered by data.')
        return(False)

    return(True)


def get_window(stats, g_speed, params):
    """
    Obtain a window centered at distance * g_speed
    stats: obspy Stats object
    params: dictionary containing 'hw' halfwidth in seconds,
    'sep_noise' separation of noise window in fraction/multiples of halfwidth,
    'wtype' window type (None,boxcar, hanning),
    'overlap' may signal windows overlap (Boolean)
    """

    # Properties of trace
    s_0 = int((stats.npts - 1) / 2)
    dist = stats.sac.dist
    Fs = stats.sampling_rate
    n = stats.npts

    # Find indices for window bounds
    ind_lo = int((dist / g_speed - params['hw']) * Fs) + s_0
    ind_hi = int((dist / g_speed + params['hw']) * Fs) + s_0
    ind_lo_n = ind_hi + int(params['sep_noise'] * params['hw'] * Fs)
    ind_hi_n = ind_lo_n + int(2 * params['hw'] * Fs)

    # Checks..overlap, out of bounds
    scs = window_checks(ind_lo, ind_hi, ind_lo_n,
                        ind_hi_n, n, params['win_overlap'])

    if scs:
        # Fill signal window
        win_signal = window(params['wtype'], n, ind_lo, ind_hi)
        # Fill noise window
        win_noise = window(params['wtype'], n, ind_lo_n, ind_hi_n)

        return win_signal, win_noise, scs

    else:
        return np.zeros(n), np.zeros(n), scs


def window(wtype, n, i0, i1):
    win = np.zeros(n)

    if wtype is None:
        win += 1.
    elif wtype == 'boxcar':
        win[i0:i1] += 1
    elif wtype == 'hann':
        win[i0: i1] += np.hanning(i1 - i0)
    else:
        msg = ('Window type \'%s\' is not implemented\nImplemented types:\
 boxcar, hann' % wtype)
        raise NotImplementedError(msg)
    return win


def snratio(correlation, g_speed, window_params):

    window = get_window(correlation.stats, g_speed, window_params)

    if not window_params['causal_side']:
        win_s = window[0][::-1]
        win_n = window[1][::-1]
    else:
        win_s = window[0]
        win_n = window[1]

    if window[2]:
        signl = np.sum((win_s * correlation.data) ** 2)
        noise = np.sum((win_n * correlation.data) ** 2)

        snr = signl / (noise + np.finfo(noise).tiny)

    else:
        snr = np.nan
    return snr
