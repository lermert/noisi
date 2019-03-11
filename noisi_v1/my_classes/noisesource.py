import numpy as np
import h5py

from scipy.stats import linregress
import os
try:
    from noisi_v1.util.plot import plot_grid
except ImportError:
    print('Plotting unavailable, is basemap installed?')

from noisi_v1.util.geo import get_spherical_surface_elements

class NoiseSource(object):
    """
   'model' of the noise source that comes in terms of a couple of basis 
    functions and associated weights. The NoiseSource object should contain a 
    function to determine weights from a (kernel? source model?), and to expand from weights and basis 
    functions.
    
    """
    
    
    def __init__(self,model,w='r'):
            
        # Model is an hdf5 file which contains the basis and weights of the source model!
        
       
        try:
            self.model = h5py.File(model,w)
            self.src_loc = self.model['coordinates']
            self.freq = self.model['frequencies']
             
            # Presumably, these arrays are small and will be used very often --> good to have in memory.
            self.distr_basis = self.model['distr_basis'][:]
            self.spect_basis = self.model['spect_basis'][:]
            self.distr_weights = self.model['distr_weights'][:]

            # The surface area of each grid element...new since June 18
            try:
                self.surf_area = self.model['surf_areas'][:]
            except KeyError:
                # approximate as spherical surface elements...
                self.surf_area = get_spherical_surface_elements(
                                            self.src_loc[0],self.src_loc[1])
                np.save('surface_areas_grid.npy',self.surf_area)
            
            self.spatial_source_model = self.expand_distr()
            
        except IOError:
            msg = 'Unable to open model file '+model
            raise IOError(msg)


        
    def __enter__(self):
        return self
    
    def __exit__(self,type,value,traceback):
        
        if self.model is not None:
            self.model.close()
            #ToDo: Check what parameters/data should be written before file closed

    def project_gridded(self):
        pass

    def expand_distr(self):
        expand = np.dot(self.distr_weights,self.distr_basis)
        
        return np.array(expand,ndmin=2)


    def get_spect(self,iloc):
        # return one spectrum in location with index iloc
        # The reason this function is for one spectrum only is that the entire gridded matrix of spectra by location is most probably pretty big.
        

        weights = self.spatial_source_model[:,iloc]#np.array(self.expand_distr()[:,iloc])
        
        
        return np.dot(weights, self.spect_basis)
    
    
    def plot(self,**options):
        
        # plot the distribution
       
        for m in self.spatial_source_model: 
            plot_grid(self.src_loc[0],self.src_loc[1],m,**options)


    
    # Note: Inefficient way of doing things! Whichever script needs the noise source field should rather look up things directly in the hdf5 file.
    # But: ToDo: This could be used internally to write to a file, rather than reading from.
    # Although: A problem to think about: noise source should behave the same, whether it is defined by model or by file. So maybe, since model will be the default option anyway, work with this!
    #def get_spectrum(self,iloc):
    #    # Return the source spectrum at location nr. iloc
    #    
    #    #if self.file is not None:
    #    #    return self.sourcedistr[iloc] * self.spectra[iloc,:]
    #    if self.model is not None:
    #        return self.spectr.
    #        # (Expand from basis fct. in model)
    #def get_sourcedistr(self,i):
    #    # Return the spatial distribution of max. PSD
    #    if self.file is not None:
    #        return np.multiply(self.sourcedistr[:],self.spectra[:,i])
    #    else:
    #        raise NotImplementedError
   
