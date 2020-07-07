#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name = 'noisi',
    version = '0.1.0a0',
    description = 'Package to calculate noise correlations from precomputed\
 seismic wavefields',
 long_description = "Python package to simulate ambient seismic noise cross-correlations and sensitivity kernels to noise sources for noise source optimization. Earth's oceans, wind, human activity and other sources create ambient seismic vibrations that are picked up by seismometers and used by seismologists to study the interior of the Earth and the interactions of solid Earth, oceans and atmosphere. The noisi tool provides a framework for the convenient simulation of auto- and cross-correlations of the ambient noise. Find out more at: https://github.com/lermert/noisi, https://github.com/jigel/noisi_inv",
    url = 'https://github.com/lermert/noisi', 
    author = 'L. Ermert, J. Igel, A. Fichtner',
    author_email  = 'lermert@fas.harvard.edu, jonas.igel@erdw.ethz.ch',
    license = 'GNU Lesser General Public License, Version 3 (LGPLv3) or later',
    classifiers = [
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Environment :: Console'
    ],
    keywords = 'Ambient seismic noise',
    packages = find_packages(),
    package_data={'noisi':['config/config.yml',
                  'config/source_config.yml',
                  'config/measr_config.yml',
                  'config/config_comments.txt',
                  'config/measr_config_comments.txt',
                  'config/source_setup_parameters.yml',
                  'config/stationlist.csv',
                  'config/data_sac_headers.txt',
                  'config/source_config_comments.txt']},
    install_requires = [
        "numpy",
        "scipy",
        "obspy>=1.0.1",
        "geographiclib",
        "mpi4py>=2.0.0",
        "pandas",
        "h5py",
        "PyYaml",
        "jupyter",
        "pytest"],
    entry_points = {
        'console_scripts': [
            'noisi = noisi.main:run'
        ]
    },
)

