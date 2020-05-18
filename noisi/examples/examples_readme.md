## Examples
===========

1. noisi_introduction.ipynb
2. optimize_noise_source_model.py
3. wavefield_from_ax3d.py

These examples are intended to facilitate the usage of the noisi tool. After installation, we recommend to open the tutorial noisi_introduction.ipynb in a jupyter notebook. This will walk you through all major steps of the modelling procedure.

Secondly, the jupyter notebook tutorial will create an example setup that can subsequently be used to run an inversion:
The script optimize_noise_source_model.py can be used for inversion. The example in optimize_noise_source_model.py is intended as a minimal working example that does not produce any meaningful model, but illustrates all the steps of the algorithm and does not need a long time to complete. If you intend to run larger examples making use of mpi4py, feel free to contact us as this requires tweaking the optimization algorithm of scipy.

Finally, in order to use a wavefield simulated with axisem3d in noisi, you can modify and use the script wavefield_from_ax3d to create a wavefield for noisi on the basis of an axisem surface wave field output. Note that axisem3d output is not included even for illustrative purposes due to the large file sizes. To create such output, head to the [axisem3d repository](https://github.com/kuangdai/AxiSEM3D) and follow installation instructions.
To make use of reciprocity, you can run the script stationlist_to_CMTSOLUTIONS.py which will create CMTSOLUTION files for point force sources in axisem3d. Please  Then, you can follow two strategies:
a) save the surface wavefield of axisem3d, and use script wavefield_from_ax3d.py to extract it at the source locations defined in the noisi setup (more convenient)
b) create a STATIONS file by using sourcegrid_to_stationlist.py, and extract from the seismograms saved by axisem3d using the script convert_ax3d_hdf.py (less convenient, but sometimes necessary if the surface wavefield is too heavy to write to disk in one single file).