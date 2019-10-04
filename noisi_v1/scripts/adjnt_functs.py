import numpy as np
from noisi_v1.util import windows as wn
from scipy.signal import hilbert


def log_en_ratio_adj(corr_o, corr_s, g_speed, window_params):

    success = False
    window = wn.get_window(corr_o.stats, g_speed, window_params)
    win = window[0]

    wn.my_centered(corr_s.data, corr_o.stats.npts)

    if window[2]:
        sig_c = corr_s.data * win
        sig_a = corr_s.data * win[::-1]
        E_plus = np.trapz(np.power(sig_c, 2)) * corr_s.stats.delta
        E_minus = np.trapz(np.power(sig_a, 2)) * corr_s.stats.delta
        # to win**2
        u_plus = sig_c * win
        u_minus = sig_a * win[::-1]
        adjt_src = 2. * (u_plus / E_plus - u_minus / E_minus)
        success = True
    else:
        adjt_src = win - win + np.nan
    return adjt_src, success


def windowed_waveform(corr_o, corr_s, g_speed, window_params):

    success = False
    window = wn.get_window(corr_o.stats, g_speed, window_params)
    win = window[0] + window[0][::-1]
    if window[2]:

        u_s = np.multiply(win, corr_s.data)
        u_o = np.multiply(win, corr_o.data)

        adjt_src = np.multiply(win, (u_s - u_o))
        success = True
    else:
        adjt_src = win - win + np.nan

    return adjt_src, success


def full_waveform(corr_o, corr_s, **kwargs):
    adjt_src = corr_s.data - corr_o.data
    return adjt_src, 1


def square_envelope(corr_o, corr_s, g_speed,
                    window_params):
    success = False
    env_s = corr_s.data**2 + np.imag(hilbert(corr_s.data))**2
    env_o = corr_o.data**2 + np.imag(hilbert(corr_o.data))**2
    d_env_1 = corr_s.data
    d_env_2 = (np.imag(hilbert(corr_s.data)))

    u1 = (env_s - env_o) * d_env_1
    u2 = np.imag(hilbert((env_s - env_o) * d_env_2))

    adjt_src = 2 * (u1 - u2)

    success = True
    return adjt_src, success


def energy(corr_o, corr_s, g_speed, window_params):

    success = False
    window = wn.get_window(corr_o.stats, g_speed, window_params)

    win = window[0]
    if window[2]:
        u1 = 2 * np.multiply(np.power(win, 2), corr_s.data)
        u2 = 2 * np.multiply(np.power(win[::-1], 2), corr_s.data)
        adjt_src = [u1, u2]
        success = True
    else:
        adjt_src = [win - win + np.nan, win - win + np.nan]

    return adjt_src, success


def get_adj_func(mtype):
    if mtype == 'ln_energy_ratio':
        func = log_en_ratio_adj

    elif mtype == 'energy_diff':
        func = energy

    elif mtype == 'windowed_waveform':
        func = windowed_waveform
    elif mtype == 'full_waveform':
        func = full_waveform

    elif mtype == 'square_envelope':
        func = square_envelope

    else:
        msg = 'Measurement functional %s not currently implemented. \
Must be one of ln_energy_ratio, energy_diff, windowed_waveform.' % mtype
        raise ValueError(msg)
    return func
