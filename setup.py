#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name = 'noisi',
    version = '0.0.0a0',
    description = 'Package to calculate noise correlations from precomputed\
 seismic wavefields',
    #long_description =
    # url = 
    author = 'L. Ermert, J. Igel, A. Fichtner',
    author_email  = 'laura.ermert@earth.ox.ac.uk, jigel@student.ethz.ch',
    # license
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Topic :: Seismology',
        'Programming Language :: Python :: 3',
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
        "cartopy",
        "jupyter",
        "pytest"],
    entry_points = {
        'console_scripts': [
            'noisi = noisi.main:run'
        ]
    },
)

