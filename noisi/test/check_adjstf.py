import numpy as np
import matplotlib.pyplot as plt
from obspy import Trace
from noisi.scripts import measurements as rm
from noisi.scripts import adjnt_functs as af

# more or less replicating Korbi's test with my measurement and adjoint source
# *********************************************
# input:
# *********************************************
steps = np.arange(-14, 0, 0.1)
mtype = 'full_waveform'
sacdict = {'dist': 1e6}
g_speed = 3700.
window_params = {}
window_params['hw'] = 200
window_params['sep_noise'] = 1.
window_params['win_overlap'] = False
window_params['wtype'] = 'hann'
window_params['causal_side'] = False
window_params['plot'] = False
# *********************************************
# *********************************************

m_a_options = {'g_speed': g_speed, 'window_params': window_params}
m_func = rm.get_measure_func(mtype)
a_func = af.get_adj_func(mtype)


# create observed data, synthetics and perturbation
c_obs = 2 * (np.random.rand(2401,) - 0.5)
c_ini = 2 * (np.random.rand(2401,) - 0.5)
d_c = 2 * (np.random.rand(2401,) - 0.5)

c_obs = Trace(data=c_obs)
c_obs.stats.sampling_rate = 1.0
c_obs.stats.sac = sacdict

c_syn = Trace(data=c_ini)
c_syn.stats.sampling_rate = 1.0
c_syn.stats.sac = sacdict

# obtain a measurement and an adjoint source time function
# for the unperturbed measurement
msr_o = m_func(c_obs, **m_a_options)
msr_s = m_func(c_syn, **m_a_options)
data, success = a_func(c_obs, c_syn, **m_a_options)


if mtype == 'energy_diff':
    data = data[0] + data[1]
    msr_s = msr_s[0] + msr_s[1]
    msr_o = msr_o[0] + msr_o[1]
    data *= (msr_s - msr_o)

elif mtype == 'ln_energy_ratio':
    data *= (msr_s - msr_o)

elif mtype in ['windowed_waveform', 'full_waveform']:
    pass

if mtype in ['ln_energy_ratio', 'energy_diff']:
    j = 0.5 * (msr_s - msr_o)**2
elif mtype in ['full_waveform', 'windowed_waveform', 'square_envelope', 'envelope']:
    j = 0.5 * np.sum(np.power((msr_s - msr_o), 2))

# left hand side of test 1:
# adjt source time function * du = change of misfit wrt u
djdc = np.dot(data, d_c)

# right hand side of test 1:
# Finite difference approx of misfit change for different steps
dcheck = []
d_ch = c_syn.copy()


for step in steps:
    d_ch.data = c_ini + 10. ** step * d_c
    msr_sh = m_func(d_ch, **m_a_options)
    if mtype == 'energy_diff':
        msr_sh = msr_sh[0] + msr_sh[1]

    jh = 0.5 * (msr_sh - msr_o)**2
    if mtype in ['full_waveform', 'windowed_waveform', 'envelope', 'square_envelope']:
        jh = 0.5 * np.sum(np.power((msr_sh - msr_o), 2))

    djdch = (jh - j) / (10.**step)
    dcheck.append(abs(djdc - djdch) / abs(djdc))


# plot
plt.semilogy(steps, dcheck)
plt.title("Check for adjoint source time function")
plt.show()
