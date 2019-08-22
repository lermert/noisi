import argparse
from mpi4py import MPI
from noisi_v1.scripts.source_grid import setup_sourcegrid as setup_sgrid
from noisi_v1.util.setup_new import setup_proj
# from noisi_v1.scripts.run_correlation import run_corr
from noisi_v1.scripts.correlation import run_corr
from noisi_v1.scripts.run_wavefieldprep import precomp_wavefield
from noisi_v1.scripts.run_sourcesetup import source_setup
from noisi_v1.scripts.run_measurement import run_measurement
from noisi_v1.scripts.kernel import run_kern

# simple embarrassingly parallel run:
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def hello(args, comm, size, rank):
    if rank == 0:
        print('Noisi Version 1. Type noisi -h for more information.\n')


parser = argparse.ArgumentParser(description='Noise cross-correlation tool.')
parser.set_defaults(func=hello)
subparsers = parser.add_subparsers()

###########################################################################
# Setting up a new project
###########################################################################
parser_setup_project = subparsers.add_parser('setup_project',
                                             help='Initialize a new project.')
parser_setup_project.add_argument('project_name', type=str, help='Project name. Noisi will\
 create a new directory under this name. The directory must not exist yet.')
parser_setup_project.set_defaults(func=setup_proj)

# ###########################################################################
# Setting up the discretized source grid
# ###########################################################################

parser_setup_sourcegrid = subparsers.add_parser('setup_sourcegrid',
                                                help='Set up the discretized\
 source grid.')
parser_setup_sourcegrid.add_argument('project_path', type=str,
                                     help='path to project directory')
parser_setup_sourcegrid.set_defaults(func=setup_sgrid)

# ###########################################################################
# Prepare the wave field
# ###########################################################################


def precompute_wavefield(args, comm, size, rank):
    pw = precomp_wavefield(args, comm, size, rank)
    pw.precompute()


parser_setup_wavefield = subparsers.add_parser('setup_wavefield',
                                               help='Set up pre-computed \
Green\'s function database.')
parser_setup_wavefield.add_argument('project_path', type=str,
                                    help='path to project directory')
parser_setup_wavefield.add_argument('--v', type=float, dest='v',
                                    help='surface wave phase velocity in mps',
                                    default=3000.0)
parser_setup_wavefield.add_argument('--q', type=float, dest='q',
                                    help='surface wave quality factor',
                                    default=100.0)
parser_setup_wavefield.add_argument('--rho', type=float, dest='rho',
                                    help='medium density in kg / m^3',
                                    default=3000.0)
parser_setup_wavefield.set_defaults(func=precompute_wavefield)

# #############################################################################
# Create source directory structure
###############################################################################

parser_setup_source_dir = subparsers.add_parser('setup_source_dir',
                                                help='Set up directory \
structure for a new source model.')
parser_setup_source_dir.add_argument('source_model', type=str, help='Name for\
 new source model. This will create a directory by that name and copy the\
 default input there. The second run will set up the source starting model as\
 specified in the configurations.')
parser_setup_source_dir.add_argument('--new_model', default=True)
parser_setup_source_dir.set_defaults(func=source_setup)


# #############################################################################
# Initialize a source model
###############################################################################

parser_setup_source = subparsers.add_parser('setup_source',
                                            help='Initialize a new source\
 model.')
parser_setup_source.add_argument('source_model', type=str, help='Name for the\
 new source model. This will create a directory by that name and copy the\
 default input there. The second run will set up the source starting model as\
 specified in the configurations.')
parser_setup_source.add_argument('--new_model', default=False)
parser_setup_source.set_defaults(func=source_setup)


# ###########################################################################
# Correlations
# ###########################################################################

parser_correlation = subparsers.add_parser('correlation',
                                           help='Calculate correlations.')
parser_correlation.add_argument('source_model', type=str, help='Path to source\
 model.')
parser_correlation.add_argument('step', type=int, help='Iteration step (start\
 at 0.)')
parser_correlation.add_argument('--steplengthrun',
                                default=False, required=False)
parser_correlation.add_argument('--ignore_network',
                                default=True, required=False)
parser_correlation.set_defaults(func=run_corr)


###########################################################################
# Measure and get adjoint sources
###########################################################################
parser_measurement = subparsers.add_parser('measurement',
                                           help='Take measurements.')
parser_measurement.add_argument('source_model', type=str, help='Path to source\
 model.')
parser_measurement.add_argument('step', type=int, help='Iteration step (start\
 at 0.)')
parser_measurement.add_argument('--steplengthrun',
                                default=False, required=False)
parser_measurement.add_argument('--ignore_network',
                                default=True, required=False)

parser_measurement.set_defaults(func=run_measurement)

# ###########################################################################
# Get kernels
# ###########################################################################
parser_kernel = subparsers.add_parser('kernel', help='Calculate source \
sensitivity kernels.')
parser_kernel.add_argument('source_model', type=str, help='Path to source\
 model.')
parser_kernel.add_argument('step', type=int, help='Iteration step (start\
 at 0.)')
parser_kernel.add_argument('--ignore_network',
                           default=True, required=False)
parser_kernel.set_defaults(func=run_kern)

# ###########################################################################
# ### Step length test forward model
# ###########################################################################
# @run.command(help='Calculate fewer correlations for step length test.')
# @click.argument('source_model')
# @click.argument('step')
# def step_test(source_model,step):
#     source_model = os.path.join(source_model,'source_config.json')
#     run_corr(source_model,step,steplengthrun=True)


def run():
    """
    Main routine for noise correlation modeling and noise source inversion.
    """
    args = parser.parse_args()
    args.func(args, comm, size, rank)
