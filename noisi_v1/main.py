# import os
# import io
# import json
# import time
import argparse

from noisi_v1.scripts.source_grid import setup_sourcegrid as setup_sgrid
from noisi_v1.util.setup_new import setup_proj, setup_source
# from noisi.scripts.run_correlation import run_corr
# from noisi.scripts.run_measurement import run_measurement
# from noisi.scripts.run_adjointsrcs import run_adjointsrcs
# from noisi.scripts.run_kernel import run_kern
# from noisi.scripts.run_preprocessing import run_preprocessing
# from noisi.scripts.run_preprocessing_data import run_preprocess_data
# from noisi.scripts.assemble_gradient import assemble_ascent_dir


def hello(args):
    print('Noisi Version 1. Type noisi_v1 -h for more information.\n')

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
parser_setup_sourcegrid.add_argument('project_path', type=str, help='path to project directory')
parser_setup_sourcegrid.set_defaults(func=setup_sgrid)

# #############################################################################
# Initialize a source model
###############################################################################

parser_setup_source = subparsers.add_parser('setup_source',
                                             help='Initialize a new source\
 model.')
parser_setup_source.add_argument('source_model', type=str, help='Name for the\
 new source model. This will create a directory by that name and copy the\
 default input there.')
parser_setup_source.set_defaults(func=setup_source)


def run():
    """
    Main routine for noise correlation modeling and noise source inversion.
    """
    args = parser.parse_args()
    args.func(args)

# ###########################################################################
# ### Preprocess the sytnthetic wavefields
# ########################################################################### 
# @run.command(help='Filter & truncate synthetics before correlation.')
# @click.argument('source_model')
# def preprocess_synthetics(source_model):
#     source_model = os.path.join(source_model,'source_config.json')
#     source_config = json.load(open(source_model))
#     if source_config['preprocess_do']:
        
#         dir = os.path.join(source_config['source_path'],'wavefield_processed')
        
#         try:
#             os.mkdir(dir)
#         except:
#             pass    
            
#         run_preprocessing(source_config)


# ###########################################################################
# ### Correlations <3
# ###########################################################################
# @run.command(help='Calculate correlations for selected source model.')
# @click.argument('source_model')
# @click.argument('step')
# def correlation(source_model,step):
#     source_model = os.path.join(source_model,'source_config.json')
#     run_corr(source_model,step)
    

# ###########################################################################
# ### Measure and get adjoint sources
# ###########################################################################
# @run.command(help='Run measurement and adjoint sources.')
# @click.argument('source_model')
# # To do: Include a --test option that produces only plots 
# # To do: include a get_parameters_options or something, so that there is no 
# # extra step necessary in run_measurement
# @click.argument('step')
# @click.option('--ignore_network',is_flag=True)
# @click.option('--step_test',is_flag=True)
# def measurement(source_model,step,ignore_network,step_test):
    
#     measr_config = os.path.join(source_model,'measr_config.json')
#     source_model = os.path.join(source_model,'source_config.json')
    
#     run_measurement(source_model,measr_config,int(step),ignore_network,
#         step_test)
#     if not step_test:
#         run_adjointsrcs(source_model,measr_config,int(step),ignore_network)


# ###########################################################################
# ### Get kernels (without residuals multiplied)
# ###########################################################################
# @run.command(help='Calculate preliminary kernels.')
# @click.argument('source_model')
# @click.argument('step')
# @click.option('--ignore_network',is_flag=True)
# def kernel(source_model,step,ignore_network):
#     source_model = os.path.join(source_model,'source_config.json')
#     run_kern(source_model,step,ignore_network=ignore_network)


# ###########################################################################
# ### Step length test forward model
# ###########################################################################
# @run.command(help='Calculate fewer correlations for step length test.')
# @click.argument('source_model')
# @click.argument('step')
# def step_test(source_model,step):
#     source_model = os.path.join(source_model,'source_config.json')
#     run_corr(source_model,step,steplengthrun=True)


# ###########################################################################
# ### Assemble the gradient by multplying kernels by residuals and summing
# ###########################################################################
# @run.command(help='Assemble ascent direction from spatial kernels and \
# measurements')
# @click.argument('source_model')
# @click.argument('step')
# @click.option('--snr_min',default=0.0)
# @click.option('--n_min',default=0)
# @click.option('--normalize',default=False)
# def gradient(source_model,step,snr_min,n_min,normalize):
#     snr_min = float(snr_min)
#     source_model = os.path.join(source_model,'source_config.json')
#     assemble_ascent_dir(source_model,step,snr_min,
#         n_min,normalize_gradient=normalize)
    


# ###########################################################################
# ### Older stuff, might be useful again but maybe not
# ###########################################################################

# ###########################################################################
# ### Old: prepare input for specfem
# ###########################################################################
# @run.command(help='Prepare specfem input.')
# @click.argument('project_path')
# def specfem_input(project_path):
#     prepare_specfem_input(os.path.join(project_path,'config.json'))


# ###########################################################################
# ### Old: Preprocess data (filtering is done anyway by measurement, if asked!)
# ########################################################################### 
# @run.command(help='Preprocess observed correlations')
# @click.argument('source_model')
# @click.option('--bandpass',help='Bandpass filter, format: freq1 freq2 corners.',
#     default=None)
# @click.option('--decimator',help='Decimation factor. Default obspy antialias \
# filter will be run before decimating.',default=None)
# @click.option('--fs_new',help='New sampling rate. Ensure that filtering is \
# performed before!',default=None)
# def preprocess_data(source_model,bandpass,decimator,fs_new):

#     if bandpass is not None:
#         bandpass = [float(bp) for bp in bandpass.split()]

#     if fs_new is not None:
#         fs_new = float(fs_new)

#     if decimator is not None:
#         decimator = int(decimator)

    
#     run_preprocess_data(source_model,bandpass=bandpass,
#         decimator=decimator,Fs_new=fs_new)
