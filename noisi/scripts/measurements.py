"""
Measurements on correlation traces used in noisi
:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np
from scipy.signal import hilbert
from noisi.util.windows import get_window
from math import log


def square_envelope(corr_o, corr_s, g_speed, window_params):
    if corr_o.data.max() != 0:
        a = abs(corr_o.data.max())
        corr_s.data /= a
        corr_o.data /= a

    se_o = corr_o.data ** 2 + np.imag(hilbert(corr_o.data)) ** 2
    se_s = corr_s.data ** 2 + np.imag(hilbert(corr_s.data)) ** 2

    return 0.5 * np.sum(corr_o.stats.delta * (se_s - se_o) ** 2)


def windowed_waveform(corr_o, corr_s, g_speed, window_params):
    window = get_window(corr_o.stats, g_speed, window_params)
    win = window[0]
    if window[2]:
        if corr_o.data.max() != 0:
            a = abs(corr_o.data.max())
            corr_s.data /= a
            corr_o.data /= a
        win_caus = (corr_o.data * win)
        win_acaus = (corr_o.data * win[::-1])
        msr_o = win_caus + win_acaus
        
        win_caus_s = (corr_s.data * win)
        win_acaus_s = (corr_s.data * win[::-1])
        msr_s = win_caus_s + win_acaus_s
    else:
        msr_o = win - win + np.nan
        msr_s = win - win + np.nan

    return 0.5 * np.sum(corr_o.stats.delta * np.power((msr_s - msr_o), 2))


def full_waveform(corr_o, corr_s, **kwargs):
    
    if corr_o.data.max() != 0:
        a = abs(corr_o.data.max())
        corr_s.data /= a
        corr_o.data /= a

    return 0.5 * np.sum(corr_o.stats.delta * np.power((corr_s.data - corr_o.data), 2))


def energy(corr_o, corr_s, g_speed, window_params):

    window = get_window(corr_o.stats, g_speed, window_params)
    msr = [np.nan, np.nan]
    win = window[0]

    if window[2]:
        if corr_o.data.max() != 0:
            a = abs(corr_o.data.max())
            corr_s.data /= a
            corr_o.data /= a
        # causal
        E_o = np.trapz(corr_o.stats.delta * (corr_o.data * win)**2)
        E_s = np.trapz(corr_o.stats.delta * (corr_s.data * win)**2)
        msr[0] = 0.5 * (E_s - E_o) ** 2

        # acausal
        win = win[::-1]
        E_o = np.trapz(corr_o.stats.delta * (corr_o.data * win)**2)
        E_s = np.trapz(corr_o.stats.delta * (corr_s.data * win)**2)
        
        msr[1] = 0.5 * (E_s - E_o) ** 2
       

    return np.array(msr)


def log_en_ratio(corr_o, corr_s, g_speed, window_params):
    delta = corr_o.stats.delta
    window = get_window(corr_o.stats, g_speed, window_params)
    win = window[0]

    if window[2]:

        # data
        sig_c = corr_o.data * win
        sig_a = corr_o.data * win[::-1]
        E_plus = np.trapz(np.power(sig_c, 2)) * delta
        E_minus = np.trapz(np.power(sig_a, 2)) * delta
        msr_o = log(E_plus / (E_minus))# + np.finfo(E_minus).tiny))
        # synthetic
        sig_c = corr_s.data * win
        sig_a = corr_s.data * win[::-1]
        E_plus = np.trapz(np.power(sig_c, 2)) * delta
        E_minus = np.trapz(np.power(sig_a, 2)) * delta
        msr_s = log(E_plus / (E_minus))# + np.finfo(E_minus).tiny))

    else:
        msr_o = np.nan
        msr_s = np.nan
    return 0.5 * (msr_s - msr_o) ** 2


# This is a bit problematic cause here the misfit already needs
# to be returned (for practical reasons) -- ToDo think about
# how to organize this better
# def inst_mf(corr_obs, corr_syn, g_speed, window_params):
#     window = get_window(corr_obs.stats, g_speed, window_params)
#     win = window[0]

#     if window[2]:

#         sig1 = corr_obs.data * (win + win[::-1])
#         sig2 = corr_syn.data * (win + win[::-1])
#     # phase misfit .. try instantaneous phase
#     # hilbert gets the analytic signal (only the name is confusing)
#         a1 = hilbert(sig1)
#         a2 = hilbert(sig2)

#         cc = a1 * np.conjugate(a2)

#         boxc = np.clip((win + win[::-1]), 0, 1)
#         dphase = 0.5 * np.trapz(np.angle(cc * boxc)**2)

#         if window_params['plot']:
#             plot_window(corr_obs, win, dphase)
#     else:
#         dphase = np.nan

#     return dphase


def envelope(corr_o, corr_s, g_speed, window_params):
    if corr_o.data.max() != 0:
        a = abs(corr_o.data.max())
        corr_s.data /= a
        corr_o.data /= a

    se_o = corr_o.data ** 2 + np.imag(hilbert(corr_o.data)) ** 2
    en_o = np.sqrt(se_o)
    se_s = corr_s.data ** 2 + np.imag(hilbert(corr_s.data)) ** 2
    en_s = np.sqrt(se_s)

    return 0.5 * np.sum(corr_o.stats.delta * (en_s - en_o) ** 2)


def get_measure_func(mtype):

    if mtype == 'ln_energy_ratio':
        func = log_en_ratio
    elif mtype == 'energy_diff':
        func = energy
    elif mtype == 'square_envelope':
        func = square_envelope
    elif mtype == 'envelope':
        func = envelope
    elif mtype == 'windowed_waveform':
        func = windowed_waveform
    elif mtype == 'full_waveform':
        func = full_waveform
    else:
        msg = 'Measurement functional %s not currently implemented.' % mtype
        raise ValueError(msg)
    return func
