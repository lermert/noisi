import numpy as np
import h5py
try:
    from noisi.util.plot import plot_grid
except ImportError:
    print('Plotting unavailable, is basemap installed?')


class NoiseSource(object):

    """
    Bookkeeping device for ambient seismic source models.
    """

    def __init__(self, model, w='r'):

        """
        :param model: hdf5 file which contains the spectral coefficients
        :param w: Allow read or write access to underlying hdf5 file
        """

        try:
            self.model = h5py.File(model, w)
            self.src_loc = self.model['coordinates']
            self.freq = self.model['frequencies']

            # Presumably, these arrays are small and will be used very often
            # --> good to have in memory.
            self.distr_basis = self.model['model'][:]
            self.spect_basis = self.model['spectral_basis'][:]

            # The surface area of each grid element
            self.surf_area = self.model['surface_areas'][:]

        except IOError:
            msg = 'Unable to open model file %r.' % model
            raise IOError(msg)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):

        if self.model is not None:
            self.model.close()

    def get_spect(self, iloc, by_index=True):

        # return one spectrum in location with index iloc
        return np.dot(self.distr_basis[iloc, :],
                      self.spect_basis)

    def plot(self, **options):

        # plot the distribution
        for i in range(self.distr_basis.shape[-1]):
            m = self.distr_basis[:, i]
            plot_grid(self.src_loc[0], self.src_loc[1], m, **options)
