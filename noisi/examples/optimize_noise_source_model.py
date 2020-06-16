"""
Sample script to optimize ambient noise source models using
scipy.optimize and noisi
:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""
from mpi4py import MPI
from scipy.optimize import minimize, Bounds
from noisi import NoiseSource
# imports of the scripts doing the actual modeling and inversion:
from noisi.scripts.correlation import run_corr
from noisi.scripts.kernel import run_kern
from noisi.scripts.run_measurement import run_measurement
# further imports
from noisi.util.smoothing import smooth
import numpy as np
from pandas import read_csv
from glob import glob
import h5py
import os
import warnings
# =============================================================================
# User input
# =============================================================================
optimization_scheme = 'L-BFGS-B'  # 'CG' for conjugate gradient
model_folder_name = 'example/source_1'
# file of the source grid belonging to this simulation
srcgridfile = 'example/sourcegrid.npy'
# terminate after this many iterations
max_iterations = 4
# Values for threshold of the gradient and objective function
# when the value is reached, iterations stop
# depending on the algorithm, gtol or ftol or tol is used;
# consult the scipy manual for documentation
tol = 1.e-12
gtol = 1.e-16
ftol = 1.e-16
ord_norm = "Inf"  # select order for gradient norm to compare gradient to gtol.
# "Inf" = maximum "-Inf" = minimum,
# 1 = L1, 2 = L2 etc.
# only for bounded L-BFGS (L-BFGS-B): 
# set bounds (useful to enforce non-negativity)
bounds = Bounds(0, 100)
# regularization type: gauss_smooth or None
regularization = {'type': "gauss_smooth"}
# parameters for gaussian smoothing
smoothing_params = {'sigma_meters': [50000], 'cap': 98, 'thresh': 0}
# smoothing type: gauss or None
# smoothing cap: cutoff at percentile
# smoothing sigma_meters: Gaussian half width in m
# if several bandpass filters are applied, here they can be weighted
# (these would be specified in the source/measr_config.json file)
# this list MUST have the same length as the number of bandpass 
# filters applied during measurement.
weights = [1.0]
do_not_print_warnings = True
measurement_type = "energy_diff"
# make sure that other measurement files do not mess up the inversion
# note that this must be set to the same measurement in measr_config.json file!
# =============================================================================
# end of input
# =============================================================================

# =============================================================================
# arguments for cross-correlation model and kernel and measurement
# =============================================================================
class iteration_args(object):
    """arguments for running the correlation computation"""

    def __init__(self):
        self.source_model = model_folder_name
        self.iteration = os.path.join(model_folder_name, 'iteration_0')
        self.step = 0
        self.steplengthrun = False
        self.ignore_network = True
        self.weights = weights
# =============================================================================


def callbackF(Xi):

    itargs.step += 1
    new_iteration = os.path.join(model_folder_name, 'iteration_' +
                                 str(itargs.step))
    if rank == 0:
        print('C - ', end='', flush=True)
        os.system('mkdir -p ' + new_iteration)
        os.system('mkdir -p ' + os.path.join(new_iteration, 'corr'))
        os.system('mkdir -p ' + os.path.join(new_iteration, 'adjt'))
        os.system('mkdir -p ' + os.path.join(new_iteration, 'kern'))
        os.system("rm -f " + os.path.join(itargs.iteration, "corr", "*"))
        os.system("rm -f " + os.path.join(itargs.iteration, "adjt", "*"))
        os.system("rm -f " + os.path.join(itargs.iteration, "kern", "*"))
        model_file_path = os.path.join(itargs.iteration,
                                       'starting_model.h5')
        os.system('cp ' + model_file_path + ' ' + new_iteration)

    itargs.iteration = new_iteration
    print('\nRank {} is now starting iteration {}'.format(rank, itargs.step))
    comm.barrier()
    return()


def objective_function(x, model_shape, weights, regularization,
                       smoothing_params, x_ref, comm, size, rank, itargs):
    comm.barrier()
    if rank == 0:
        print('OF - ', end='', flush=True)
        # objective function evaluation
        # set up model
        f = h5py.File(os.path.join(itargs.iteration,
                      'starting_model.h5'), 'r+')
        f['model'][:] = x.reshape(model_shape)
        f.close()
    comm.barrier()

# call forward model
    run_corr(itargs, comm, size, rank)
    comm.barrier()

# take measurement
    if rank == 0:
        run_measurement(itargs, comm, size, rank)
        os.system('rm ' + os.path.join(itargs.iteration, 'corr', '*'))
    comm.barrier()

# evaluate measurement
    msr_files = glob(os.path.join(itargs.iteration, measurement_type +
                                  '*.measurement.csv'))
    # read each file and multiply by weight, if more elaborate weights are
    # needed they would have to happen here
    weighted_l2 = 0.0
    for i in range(len(msr_files)):
        dat = read_csv(msr_files[i])
        dat.dropna(subset=['l2_norm'], how='all', inplace=True)
        l2 = np.mean(dat.l2_norm.values)
        weighted_l2 += l2 * weights[i]
        print(weighted_l2)
# pass back weighted L2-norm
    return(weighted_l2)


def jacobian(x, model_shape, weights, regularization, smoothing, x_ref,
             comm, size, rank, itargs):
    comm.barrier()
    if rank == 0:
        print('J - ', end='', flush=True)

    # jacobian
    # call the gradient computation
    run_kern(itargs, comm, size, rank)
    comm.barrier()

    if rank == 0:
        kernels = glob(os.path.join(itargs.iteration, 'kern', '*.npy'))
        k = np.zeros(model_shape)

        # This would be the place to apply more elaborate weights
        for kern in kernels: 
            sensitivity_kernel = np.load(kern)

            # last dim: nr. of adjoint src (dep on measurement)
            # 1 for waveform, log. energy ratio
            # 2 for energy (acausal, causal)
            if sensitivity_kernel.shape[-1] == 2:
                sensitivity_kernel = sensitivity_kernel.sum(axis=-1)
            elif sensitivity_kernel.shape[-1] == 1:
                sensitivity_kernel = sensitivity_kernel[:, 0]
            
            # weights are for different bandpass filteres
            # as specified in measr_config.json
            ix_weight = int(kern.split('.')[-2])
            w = weights[ix_weight]
            k += sensitivity_kernel.T * w
        np.save('assembled_gradient.npy', np.transpose(k))
    comm.barrier()

    reg_type = regularization['type']
    if reg_type == 'gauss_smooth':
        print("smoothing.")
        # This smoothing means we are working with an inexact gradient here.
        smooth(inputfile='assembled_gradient.npy',
               outputfile='smoothed_gradient.npy',
               coordfile=srcgridfile,
               sigma=smoothing_params['sigma_meters'],
               cap=smoothing_params['cap'],
               thresh=smoothing_params['thresh'], comm=comm, size=size,
               rank=rank)
        comm.barrier()
        grad = np.load('smoothed_gradient.npy')

    else:
        grad = np.load('assembled_gradient.npy')

# pass back the gradient as 1-D array
    return(grad.T.ravel())


# =============================================================================
# run the optimization itself
# =============================================================================
# initialize stuff

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

global itargs
itargs = iteration_args()

if do_not_print_warnings:
    warnings.filterwarnings("ignore")
# read the starting model
model_file_name = os.path.join(itargs.iteration,
                               'starting_model.h5')
os.system('cp ' + model_file_name + ' starting_model_iteration_0.h5')
with NoiseSource(model_file_name) as n0:
    # rearrange as 1-D array
    model_shape = n0.distr_basis.shape
    print("model shape ", model_shape)
    x0 = n0.distr_basis[:].ravel()
    x_ref = x0.copy()

# call the optimization
result = minimize(objective_function, x0=x0,
                  args=(model_shape, weights, regularization, smoothing_params,
                        x_ref, comm, size, rank, itargs),
                  method=optimization_scheme, bounds=bounds,
                  jac=jacobian, tol=tol, callback=callbackF,
                  options={'disp': 1, 'gtol': gtol, 'ftol': ftol,
                           'maxiter': max_iterations, 'parallel_rank': rank,
                           'itargs': itargs})

if rank == 0:
    np.save('inversion_result.npy', result.x.reshape(model_shape))
