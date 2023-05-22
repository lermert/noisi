"""
Tests for noisi
:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""
from noisi.scripts.run_wavefieldprep import precomp_wavefield
from noisi.scripts import adjnt_functs as am
from noisi.scripts import measurements as rm
from noisi.util.windows import my_centered, get_window, snratio
from noisi.util.geo import geographical_distances, is_land,\
    geograph_to_geocent, get_spherical_surface_elements, points_on_ell, wgs84,\
    len_deg_lat, len_deg_lon
import pytest
import numpy as np
from obspy import Trace
from math import floor
from noisi.scripts.correlation import config_params, get_ns, \
    add_input_files, compute_correlation
import os
from noisi.util.corr_pairs import define_correlationpairs
from noisi import NoiseSource
from obspy.signal.invsim import cosine_taper
from noisi.scripts.kernel import define_kernel_tasks, compute_kernel
from noisi.scripts.kernel import add_input_files as input_files_kernel
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_precomp_wavefield():

    class input(object):
        def __init__():
            pass

    input.v = 2000.
    input.rho = 3000.
    input.q = 100.
    input.project_path = 'test/testdata_v1'

    p = precomp_wavefield(args=input, comm=comm, rank=rank, size=size)
    spec = p.green_spec_analytic(10000.)

    assert(type(spec) == np.ndarray)
    assert(spec.dtype == complex)
    assert(len(spec) == 3601)
    assert(pytest.approx(spec[1]) == 0.0177158666914 - 0.0178711453095j)


def test_adjoint_functions():

    observed = Trace(data=np.ones(25))
    synthetic = Trace(data=np.ones(25))
    synthetic.data[0:13] *= 2.0
    synthetic.stats.sac = {}
    synthetic.stats.sac['dist'] = 6000.
    observed.stats = synthetic.stats.copy()

    window_params = {}
    window_params['hw'] = 2.0
    window_params['sep_noise'] = 1.0
    window_params['win_overlap'] = False
    window_params['wtype'] = "hann"
    window_params['plot'] = False

    g_speed = 2000.

    adj, success = am.log_en_ratio_adj(observed, synthetic, g_speed,
                                       window_params)

    assert(success)
    assert(len(adj) == 25)
    assert(pytest.approx(adj[10]) == -0.16666666666666663)

    adj, success = am.windowed_waveform(observed, synthetic, g_speed,
                                        window_params)
    assert(success)
    assert(len(adj) == 25)
    assert(pytest.approx(adj[9]) == 1.)

    func = am.get_adj_func('energy_diff')
    adj, success = func(observed, synthetic, g_speed, window_params)
    assert(success)
    assert(len(adj) == 2)
    assert(type(adj) == list)
    assert(pytest.approx(adj[1].sum()) == 6.)


def test_measurements():

    synthetic = Trace(data=np.ones(25))
    synthetic.stats.sac = {}
    synthetic.stats.sac['dist'] = 6000.

    window_params = {}
    window_params['hw'] = 2.0
    window_params['sep_noise'] = 1.0
    window_params['win_overlap'] = False
    window_params['wtype'] = "hann"
    window_params['plot'] = False

    g_speed = 2000.

    msr = rm.square_envelope(synthetic, g_speed, window_params)
    assert(type(msr) == np.ndarray)
    assert(msr.sum() == 25)

    msr = rm.energy(synthetic, g_speed, window_params)
    assert(type(msr) == np.ndarray)
    assert(len(msr) == 2)

    msr = rm.log_en_ratio(synthetic, g_speed, window_params)
    assert(type(msr) == float)
    assert(pytest.approx(msr) == 0)

    msr = rm.windowed_waveform(synthetic, g_speed, window_params)
    assert(type(msr) == np.ndarray)
    assert(len(msr) == synthetic.stats.npts)
    assert(pytest.approx(msr.sum()) == 4)


def test_points_on_ell():

    grid = points_on_ell(111000., xmin=-1., xmax=1., ymin=-1., ymax=1.)
    assert(type(grid) == list)
    assert(len(grid) == 2)
    assert(len(grid[0]) < 10)
    assert(len(grid[0]) > 2)
    assert(max(grid[0]) <= 1.)


def test_geostuff():

    grid = np.zeros((2, 5))
    location = [45.0, 45.0]

    dist = geographical_distances(grid, location)
    assert(type(dist) == np.ndarray)
    assert(pytest.approx(dist[0]) == 6662472.7182103)
    assert is_land(location, location)[0]
    assert(floor(geograph_to_geocent(location[0])) == 44)
    assert(pytest.approx(len_deg_lat(location[0])) == 111131.779)
    assert(pytest.approx(len_deg_lon(location[0])) == 78582.91976)


def test_spherical_surface_elements():
    xg, yg = np.meshgrid([0., 1., 2., 3.], [4., 5., 6., 3.])
    grid = np.array([np.ravel(xg), np.ravel(yg)])
    surfel = get_spherical_surface_elements(grid[0], grid[1])
    assert(len(surfel) == len(grid[0]))
    assert(type(surfel) == np.ndarray)
    assert(pytest.approx(surfel[0]) == 12361699247.23782)


def test_windows():

    array1 = np.zeros(7, dtype=np.int32)
    array2 = np.zeros(8, dtype=np.int32)
    array1[3] = 1
    array2[4] = 1

    assert(my_centered(array1, 5))[2] == 1
    assert(my_centered(array2, 5))[2] == 1
    assert(my_centered(array1, 9))[4] == 1
    assert(my_centered(array2, 9))[4] == 1

    tr = Trace()
    tr.stats.sac = {}
    tr.stats.sac['dist'] = 3.0
    tr.data = my_centered(array1, 15) + 1
    params = {}
    params['hw'] = 1
    params['sep_noise'] = 0
    params['win_overlap'] = True
    params['wtype'] = 'hann'
    params['causal_side'] = True
    win = get_window(tr.stats, g_speed=1.0, params=params)
    assert(len(win) == 3)
    assert(pytest.approx(win[0][10]) == 1.0)

    snr = snratio(tr, g_speed=1.0, window_params=params)
    assert(int(snr) == 1)


def test_forward_model():

    class args(object):
        def __init__(self):
            self.source_model = os.path.join('test', 'testdata_v1',
                                             'testsource_v1')
            self.step = 0
            self.steplengthrun = False,
            self.ignore_network = True
    args = args()
    all_config = config_params(args, comm, size, rank)
    assert all_config.auto_corr

    ns = get_ns(all_config)
    assert(ns[0] == 3600)
    assert(ns[1] == 7200)

    p = define_correlationpairs(all_config.source_config
                                ['project_path'],
                                all_config.auto_corr)
    assert len(p) == 3
    assert p[0][0].split()[-1] == 'STA1'

    input_files = add_input_files(p[1], all_config)[0]
    assert os.path.basename(input_files[0]) == 'NET.STA1..MXZ.h5'

    nsrc = os.path.join('test', 'testdata_v1', 'testsource_v1', 'iteration_0',
                        'starting_model.h5')
    # use a one-sided taper: The seismogram probably has a non-zero end,
    # being cut off wherever the solver stopped running.
    taper = cosine_taper(ns[0], p=0.01)
    taper[0: ns[0] // 2] = 1.0
    correlation, sta1, sta2 = compute_correlation(input_files, all_config,
                                                  NoiseSource(nsrc), ns, taper)
    corr_saved = np.load(os.path.join('test', 'testdata_v1', 'testdata',
                                      'NET.STA1..MXZ--NET.STA2..MXZ.npy'))

    assert np.allclose(correlation, corr_saved)



def test_sensitivity_kernel():

    class args(object):
        def __init__(self):
            self.source_model = os.path.join('test', 'testdata_v1',
                                             'testsource_v1')
            self.step = 0
            self.steplengthrun = False,
            self.ignore_network = True
    args = args()
    all_config = config_params(args, comm, size, rank)
    ns = get_ns(all_config)
    p = define_kernel_tasks(all_config, comm, size, rank)
    assert len(p[0]) == 3
    assert p[0][2][1].split()[-1] == 'STA2'

    input_files = input_files_kernel(p[0][1], all_config)

    nsrc = os.path.join('test', 'testdata_v1', 'testsource_v1',
                        'spectral_model.h5')
    output_file = "test"
    # use a one-sided taper: The seismogram probably has a non-zero end,
    # being cut off wherever the solver stopped running.
    taper = cosine_taper(ns[0], p=0.01)
    taper[0: ns[0] // 2] = 1.0
    kernel = compute_kernel(input_files[0], output_file, all_config,
                            NoiseSource(nsrc), ns, taper)
    np.save("newtestkernel.npy", kernel)
    saved_kernel = np.load(os.path.join('test', 'testdata_v1', 'testdata',
                                        'NET.STA1..MXZ--NET.STA2..MXZ.0.npy'))
    assert np.allclose(kernel / kernel.max(), saved_kernel / saved_kernel.max())
