## Noisi - ambient noise cross-correlation modeling and inversion

This tool can be used to simulate noise cross-correlations and sensitivity kernels to noise sources.

### Installation

Install requirements (easiest done with anaconda)
- [obspy](https://docs.obspy.org/)
- PyYaml
- pandas
- mpi4py
- geographiclib
- cartopy
- h5py
- jupyter
- pytest

Additionally, install [instaseis](http://instaseis.net/), if you plan to use it for Green's functions.
Install jupyter notebook if you intend to run the tutorial (see below).

If you encounter problems with mpi4py, try removing it and reinstalling it using pip (`pip install mpi4py`).

#### Install pre-packaged
After installing the dependencies, run 
`pip install noisi`

#### Install editable for further development
Clone the repository with git:
`git clone https://github.com/lermert/noisi.git`

Change into the `noisi_v1/` directory. Call `pip install -v -e .` here.

After installation, change to the `noisi/noisi` directory and run `pytest`. If you encounter any errors (warnings are o.k.), we'll be grateful if you let us know. 

### Getting started
To see an overview of the tool, type `noisi --help`.
A step-by-step tutorial for jupyter notebook can be found in the `noisi/noisi` directory.
Examples on how to set up an inversion and how to import a wavefield from axisem3d are found in the noisi/examples directory.


