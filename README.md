## Noisi - ambient noise cross-correlation modeling and inversion

This tool can be used to simulate noise cross-correlations and sensitivity kernels to noise sources.

### Installation

Install requirements (easiest done with anaconda)
- [instaseis](http://instaseis.net/)
- [obspy](https://docs.obspy.org/)
- cartopy
- PyYaml
- pandas
- mpi4py

Additionally, install jupyter notebook if you intend to run the tutorial (see below).

If you encounter problems with mpi4py, try reinstalling it using pip (`pip install mpi4py`).

Clone the repository with git:
`git clone https://github.com/lermert/noisi_v1.git`

Change into the `noisi_v1/` directory. Call `pip install -v` .

After installation, consider running `pytest` in the `noisi_v1/noisi_v1` directory. 

### Getting started
To see an overview of the tool, type `noisi --help`.
A step-by-step tutorial for jupyter notebook can be found in the `noisi_v1/noisi_v1` directory.

