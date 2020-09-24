"""
Class for handling noise source models in noisi
:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np
import h5py


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
            self.distr_basis = np.array(self.model['model'][:], ndmin=3)
            self.spect_basis = np.array(self.model['spectral_basis'][:])

            # The surface area of each grid element
            self.surf_area = np.array(self.model['surface_areas'][:])

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
        return np.dot(self.distr_basis[:, iloc, :],
                      self.spect_basis)

    def plot(self, **options):
        comp = {0:"E", 1: "N", 2: "Z"}
        # plot the distribution
        for i in range(self.distr_basis.shape[0]):
            for j in range(self.distr_basis.shape[-1]):

                m = self.distr_basis[i, :, j]
                if m.sum() == 0:
                    continue
                title = "Source {}-component".format(comp[i])
                plot_grid(self.src_loc[0], self.src_loc[1], m, title=title, **options)

