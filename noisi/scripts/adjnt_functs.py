"""
Adjoint sources for computing noise source sensitivity
kernels in noisi
:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np
from noisi.util import windows as wn
from scipy.signal import hilbert
from math import log

def hilbi(array_in):
    return np.imag(hilbert(array_in))


def envelope(corr_o, corr_s, g_speed, window_params, dorder=1, 
             previous_perturbation=None, scaling_factor=1.0):
    
    if corr_o.data.max() != 0:
        a = abs(corr_o.data.max())
        corr_s.data /= a
        corr_o.data /= a

    env_s = np.sqrt(corr_s.data**2 + hilbi(corr_s.data)**2)
    env_o = np.sqrt(corr_o.data**2 + hilbi(corr_o.data)**2)
    d_env_1 =  corr_s.data / a
    d_env_2 =  hilbi(corr_s.data) / a
    
    if dorder == 1:
        u1 = (env_s - env_o) / env_s * d_env_1
        u2 = hilbi((env_s - env_o) / env_s * d_env_2)
        adjt_src = u1 - u2
    
    elif dorder == 2:
        d_c1_1 = previous_perturbation
        d_c1_2 = hilbi(previous_perturbation)
        a = (env_s - env_o) / env_s
        b = env_o / env_s**3 * (d_env_1 * d_c1_1 + d_env_2 * d_c1_2)

        adjt_src = a * d_c1_1 - hilbi(a * d_c1_2) +\
        b * d_env_1 - hilbi(b * d_env_2)
    success = 1

    return adjt_src, success


def log_en_ratio_adj(corr_o, corr_s, g_speed, window_params):

    success = False
    window = wn.get_window(corr_o.stats, g_speed, window_params)
    win = window[0]
    delta = corr_o.stats.delta

    corr_s.data = wn.my_centered(corr_s.data, corr_o.stats.npts)

    if window[2]:

        # synthetic
        sig_c = corr_s.data * win
        sig_a = corr_s.data * win[::-1]
        E_plus = np.trapz(np.power(sig_c, 2)) * delta
        E_minus = np.trapz(np.power(sig_a, 2)) * delta
        # observed
        sig_c_o = corr_o.data * win
        sig_a_o = corr_o.data * win[::-1]
        E_plus_o = np.trapz(np.power(sig_c_o, 2)) * delta
        E_minus_o = np.trapz(np.power(sig_a_o, 2)) * delta
        
        # to win**2
        u_plus = sig_c * win
        u_minus = sig_a * win[::-1]
        adjt_src = 2. * (u_plus / E_plus - u_minus / E_minus) * \
                (log(E_plus / E_minus) - log(E_plus_o / E_minus_o))
        success = True
    else:
        adjt_src = win - win + np.nan
    return adjt_src, success


def windowed_waveform(corr_o, corr_s, g_speed, window_params):

    success = False
    window = wn.get_window(corr_o.stats, g_speed, window_params)
    win = window[0] + window[0][::-1]
    if window[2]:

        if corr_o.data.max() != 0:
            a = abs(corr_o.data.max())
            corr_s.data /= (a ** 2)
            corr_o.data /= (a ** 2)

        u_s = np.multiply(win, corr_s.data)
        u_o = np.multiply(win, corr_o.data)

        adjt_src = np.multiply(win, (u_s - u_o))
        success = True
    else:
        adjt_src = win - win + np.nan

    return adjt_src, success


def full_waveform(corr_o, corr_s, **kwargs):

    if corr_o.data.max() != 0:
        a = abs(corr_o.data.max())
        corr_s.data /= (a ** 2)
        corr_o.data /= (a ** 2)

    adjt_src = corr_s.data - corr_o.data
    return adjt_src, 1


def square_envelope(corr_o, corr_s, g_speed,
                    window_params):
    success = False
    if corr_o.data.max() != 0:
        a = abs(corr_o.data.max())
        corr_s.data /= a
        corr_o.data /= a
    env_s = corr_s.data**2 + np.imag(hilbert(corr_s.data))**2
    env_o = corr_o.data**2 + np.imag(hilbert(corr_o.data))**2
    d_env_1 = corr_s.data / a
    d_env_2 = (np.imag(hilbert(corr_s.data))) / a

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

        if corr_o.data.max() != 0:
            a = abs(corr_o.data.max())
            corr_s.data /= (a)
            corr_o.data /= (a)

        # causal
        E_o_c = np.trapz(corr_o.stats.delta * (corr_o.data * win)**2)
        E_s_c = np.trapz(corr_o.stats.delta * (corr_s.data * win)**2)

        # acausal
        E_o_a = np.trapz(corr_o.stats.delta * (corr_o.data * win[:: -1])**2)
        E_s_a = np.trapz(corr_o.stats.delta * (corr_s.data * win[:: -1])**2)

        u1 = 2 * np.multiply(np.power(win, 2), corr_s.data / a) * (E_s_c - E_o_c)
        u2 = 2 * np.multiply(np.power(win[::-1], 2), corr_s.data / a) * (E_s_a - E_o_a)
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

    elif mtype == 'envelope':
        func = envelope

    else:
        msg = 'Measurement functional %s not currently implemented. \
Must be one of ln_energy_ratio, energy_diff, windowed_waveform.' % mtype
        raise ValueError(msg)
    return func
