import numpy as np
import matplotlib.pyplot as plt
from obspy import Trace, read
from obspy.signal.invsim import cosine_taper
from noisi.scripts import measurements as rm
from noisi import NoiseSource
from noisi.scripts.correlation import get_ns, config_params,\
    compute_correlation
import os
import h5py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# necessary data must be located test directory.

# *********************************************
# input:
# *********************************************
steps = np.arange(-6, 1, 0.1)
mtype = 'full_waveform' #'ln_energy_ratio'
g_speed = 3000.
window_params = {}
window_params['bandpass'] = None
window_params['hw'] = 20
window_params['sep_noise'] = 0.
window_params['win_overlap'] = False
window_params['wtype'] = 'hann'
window_params['causal_side'] = True
window_params['plot'] = False

synfile = 'test/testdata_v1/testdata/NET.STA1..MXZ--NET.STA2..MXZ.sac'
obsfile = 'test/testdata_v1/testsource_v1/observed_correlations/NET.STA1..MXZ--NET.STA2..MXZ.sac'
# unperturbed gradient
grad = np.load('test/testdata_v1/testdata/NET.STA1..MXZ--NET.STA2..MXZ.0.npy')
input_files = ['test/testdata_v1/greens/NET.STA1..MXZ.h5',
'test/testdata_v1/greens/NET.STA2..MXZ.h5']
source_file = 'test/testdata_v1/testsource_v1/iteration_0/starting_model.h5'
# *********************************************
# *********************************************


class args(object):
    def __init__(self):
        self.source_model = os.path.join('test',
                                         'testdata_v1', 'testsource_v1')
        self.step = 0
        self.steplengthrun = False,
        self.ignore_network = True


args = args()
all_config = config_params(args, comm, size, rank)
m_a_options = {'g_speed': g_speed, 'window_params': window_params}
m_func = rm.get_measure_func(mtype)
all_ns = get_ns(all_config)
ntime, n, n_corr, Fs = all_ns
taper = cosine_taper(ntime, p=0.01)
taper[0: ntime // 2] = 1.0

with NoiseSource(source_file) as n:
    surf_area = n.surf_area

# sum the dimension that relates to filters
grad = grad.sum(axis=1)

# apply surface elements for integration
for j in range(grad.shape[0]):
    grad[j, :, 0] *= surf_area

# measurement
obs = read(obsfile)[0]
syn = read(synfile)[0]
syn.stats.sac = {}
syn.stats.sac['dist'] = obs.stats.sac['dist']
msr_o = m_func(obs, **m_a_options)
msr_s = m_func(syn, **m_a_options)

# unperturbed misfit
if mtype in ['ln_energy_ratio']:
    j = 0.5 * (msr_s - msr_o) ** 2
elif mtype in ['full_waveform']:
    j = 0.5 * np.sum(np.power((msr_s - msr_o), 2))

# create perturbation
d_q_0 = 2 * (np.random.rand(grad.shape[1]) - 0.5)
d_q_1 = 2 * (np.random.rand(grad.shape[1]) - 0.5)

# left hand side of test 3: gradient * dq = change of misfit wrt q
grad_dq_0 = np.dot(grad[0, :, 0], d_q_0)
grad_dq_1 = np.dot(grad[1, :, 0], d_q_1)
grad_dq = grad_dq_0 + grad_dq_1

dcheck = []
# loop:
for step in steps:
    # add perturbation to archived model --> current model
    os.system('cp {} temp.h5'.format(source_file))
    n = h5py.File('temp.h5', 'r+')

    n['model'][:, 0] += 10. ** step * d_q_0
    n['model'][:, 1] += 10. ** step * d_q_1
    n.flush()
    n.close()
    with NoiseSource('temp.h5') as nsrc:
        correlation = compute_correlation(input_files, all_config, nsrc,
                                          all_ns, taper)

# evaluate misfit and add to list.
    syntest = Trace(data=correlation[0])
    syntest.stats.sac = {}
    syntest.stats.sac['dist'] = obs.stats.sac['dist']
    syntest.write('temp.sac', format='SAC')
    syntest = read('temp.sac')[0]
    msr_sh = m_func(syntest, **m_a_options)

    # plt.plot(correlation[0])
    # plt.plot(syn.data, '--')
    # plt.title(str(step))
    # plt.show()
    if mtype in ['ln_energy_ratio']:
        jh = 0.5 * (msr_sh - msr_o) ** 2
    elif mtype in ['full_waveform']:
        jh = 0.5 * np.sum(np.power((msr_sh - msr_o), 2))
    djdqh = (jh - j) / (10. ** step)
    print(djdqh, grad_dq)
    dcheck.append(abs(grad_dq - djdqh) / abs(grad_dq))

# plot
plt.semilogy(steps, dcheck)
plt.title("Check for gradient")
plt.show()

# clean up...
os.system('rm temp.h5')
os.system('rm temp.sac')
