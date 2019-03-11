from __future__ import print_function
import numpy as np
import os
import h5py
#from obspy import Trace
try:
    from noisi.util import plot
except ImportError:
    print('Plotting unavailable, is basemap installed?')
from noisi_v1.util import filter
try:
    from scipy.signal import sosfilt
except ImportError:
    from obspy.signal._sosfilt import _sosfilt as sosfilt
try:
    from scipy.fftpack import next_fast_len
except ImportError:
    from noisi_v1.borrowed_functions.scipy_next_fast_len import next_fast_len
#from scipy.signal.signaltools import _next_regular
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import integer_decimation
import click
from warnings import warn

#ToDo: Think about having a frequency domain field as well, maybe with keyword 'fd'    
#ToDo: Think about getting an entire wavefield into memory
#ToDo: Think about how to write stats and sourcegrid to disk at systematic points in the code.
class WaveField(object):
    """
    Object to handle database of stored wavefields.
    Basically, just a few methods to work on wavefields stored in an hdf5 file. 
    The stored seismograms have to have sampling rate as attribute Fs and number of time steps as attribute ntime; They have to have an ID of format net.sta.loc.cha    
    
    #ToDo: Docs
    """
    #ToDo stats entry real or complex data
    def __init__(self,file,sourcegrid=None,complex=False,w='r'):
    #shape=None,outfile=None,complex=False):
        
        #if file is not None: 
        self.w = w
        self.complex = complex
        
        try:   
            self.file = h5py.File(file, self.w)
        except IOError:
            msg = 'Unable to open input file ' + file
            raise IOError(msg)
        #else:
        #    try:   
        #        self.file = h5py.File(file, 'r+')
        #    except IOError:
        #        msg = 'Unable to open input file ' + file
        #        raise IOError(msg)
        #        
        
        self.stats = dict(self.file['stats'].attrs)
        #ToDo include in stats
        #self.complex = self.stats['complex']
        self.sourcegrid = self.file['sourcegrid']
        
        if self.complex:
            self.data_r = self.file['real']
            self.data_i = self.file['imag']
        else:
            self.data = self.file['data']
               
        print(self.file)
        
        #ToDo handle complex
   # Thought about using a class method here, but need a copy of the stats!
    def copy_setup(self,newfile,nt=None,ntraces=None,complex=None,w='r+'):
        
        if complex is None:
            complex = self.complex
        # Copy the stats and sourcegrid to a new file with empty (all-zero) arrays for seismograms
        
        # Shape of the new array:
        shape = list(np.shape(self.data))
        if ntraces is not None:
            shape[0] = ntraces
        if nt is not None:
            shape[1] = nt
        shape = tuple(shape)
        
        # Create new file
        file = h5py.File(newfile, 'w-')
       
        # Copy metadata
        stats = file.create_dataset('stats',data=(0,))
        for (key,value) in self.stats.items():
            file['stats'].attrs[key] = value
        
        # Ensure that nt is kept as requested
        if nt is not None and nt != self.stats['nt']:
            file['stats'].attrs['nt'] = nt

        #stats.attrs['reference_station'] = self.stats['refstation']
        #stats.attrs['data_quantity'] = self.stats['data_quantity']
        #stats.attrs['ntraces'] = shape[0]
        #stats.attrs['Fs'] = self.stats['Fs']
        #stats.attrs['nt'] = shape[1]
        
        file.create_dataset('sourcegrid',data=self.sourcegrid[:].copy()) 
        
        # Initialize data arrays
        if complex:
            file.create_dataset('real',shape,dtype=np.float32)
            file.create_dataset('imag',shape,dtype=np.float32)   
        else:
            file.create_dataset('data',shape,dtype=np.float32)
        
        print('Copied setup of '+self.file.filename)
        file.close()
         
        return(WaveField(newfile,w=w,complex=complex))
    
    #def copy_setup_real_to_complex(self,newfile,w='r+'):
    #    #Copy the stats and sourcegrid to a new file with empty (all-zero) arrays for seismograms
    #    #extend seismograms to spectra to fit the expected length of zero-padded FFT, and add real as well as imag. part
    #    file = h5py.File(newfile, 'w')
    #    file.create_dataset('stats',data=(0,))
    #    for (key,value) in self.stats.items():
    #        file['stats'].attrs[key] = value
    #    nfft = _next_regular(2*self.stats['nt']-1)
    #    shape = (self.stats['ntraces'],nfft//2+1)
    #    file.create_dataset('sourcegrid',data=self.sourcegrid[:].copy())    
    #    file.create_dataset('real',shape,dtype=np.float32)
    #    file.create_dataset('imag',shape,dtype=np.float32)
    #    
    #    file.close()
    #    return WaveField(newfile,complex=True,w=w)
    #
    
    def truncate(self,newfile,truncate_after_seconds):
        
        nt_new = int(round(truncate_after_seconds * self.stats['Fs']))
    
        with self.copy_setup(newfile,nt=nt_new) as wf:
        
            for i in range(self.stats['ntraces']):
                if self.complex:
                    wf.data_i[i,:] = self.data_i[i,0:nt_new].copy()
                    wf.data_r[i,:] = self.data_r[i,0:nt_new].copy()
                else:
                    wf.data[i,:] = self.data[i,0:nt_new].copy()
        
        #wf.file.close()
    
    def filter_all(self,type,overwrite=False,zerophase=True,outfile=None,**kwargs):
        
        if type == 'bandpass':
            sos = filter.bandpass(df=self.stats['Fs'],**kwargs)
        elif type == 'lowpass':
            sos = filter.lowpass(df=self.stats['Fs'],**kwargs)
        elif type == 'highpass':
            sos = filter.highpass(df=self.stats['Fs'],**kwargs)
        else:
            msg = 'Filter %s is not implemented, implemented filters: bandpass, highpass,lowpass' %type
            raise ValueError(msg)
        
        if not overwrite:
            # Create a new hdf5 file of the same shape
            newfile = self.copy_setup(newfile=outfile)
        else:
            # Call self.file newfile
            newfile = self#.file
        
        with click.progressbar(range(self.stats['ntraces']),label='Filtering..' ) as ind:
            for i in ind:
                # Filter each trace
                if zerophase:
                    firstpass = sosfilt(sos, self.data[i,:]) # Read in any case from self.data
                    newfile.data[i,:] = sosfilt(sos,firstpass[::-1])[::-1] # then assign to newfile, which might be self.file
                else:
                    newfile.data[i,:] = sosfilt(sos,self.data[i,:])
                # flush?
                
        if not overwrite:
           print('Processed traces written to file %s, file closed, \
                  reopen to read / modify.' %newfile.file.filename)
           
           newfile.file.close()
            

    def decimate(self,decimation_factor,outfile,taper_width=0.005):
        """
        Decimate the wavefield and save to a new file 
        """
        
        fs_old = self.stats['Fs']
        freq = self.stats['Fs'] * 0.4 / float(decimation_factor)

        # Get filter coeff
        sos = filter.cheby2_lowpass(fs_old,freq)

        # figure out new length
        temp_trace = integer_decimation(self.data[0,:], decimation_factor)
        n = len(temp_trace)
       

        # Get taper
        # The default taper is very narrow, because it is expected that the traces are very long.
        taper = cosine_taper(self.stats['nt'],p=taper_width)

       
        # Need a new file, because the length changes.
        with self.copy_setup(newfile=outfile,nt=n) as newfile:

            for i in range(self.stats['ntraces']):
                
                temp_trace = sosfilt(sos,taper*self.data[i,:])
                newfile.data[i,:] = integer_decimation(temp_trace, decimation_factor)
            
        
            newfile.stats['Fs'] = fs_old / float(decimation_factor)



    # def space_integral(self,weights=None):
    #     # ToDo: have this checked; including spatial sampling!
    #     # ToDo: Figure out how to assign the metadata...buh
    #     trace = Trace()
    #     trace.stats.sampling_rate = self.stats['Fs']
        
    #     # ToDo: Thinking about weights
    #     if not self.complex:
    #         if weights: 
    #             trace.data = np.trapz(np.multiply(self.data[:],weights[:]),axis=0)
    #         else:
    #             trace.data = np.trapz(self.data[:],axis=0)
    #     #oDo complex wavefield
    #     else:
    #         if weights: 
    #             trace.data_i = np.trapz(np.multiply(self.data_i[:],weights[:]),axis=0)
    #             trace.data_r = np.trapz(np.multiply(self.data_r[:],weights[:]),axis=0)
    #         else:
    #             trace.data_i = np.trapz(self.data_i[:],axis=0)
    #             trace.data_r = np.trapz(self.data_r[:],axis=0)
            
    #     return trace
            
    
    def get_snapshot(self,t,resolution=1):
        
        #ToDo: Ask someone who knows h5py well how to do this in a nice way!
        t_sample = int(round(self.stats['Fs'] * t))
        if t_sample >= np.shape(self.data)[1]:
            warn('Requested sample is out of bounds, resetting to last sample.')
            t_sample = np.shape(self.data)[1]-1
        if resolution == 1:
            snapshot = self.data[:,t_sample]
        else:
            snapshot = self.data[0::resolution,t_sample] #0:len(self.data[:,0]):resolution
        print('Got snapshot')
        
        return snapshot
    
    #ToDo put somewhere else    
    def plot_snapshot(self,t,resolution=1,**kwargs):
        
        if self.sourcegrid is None:
            msg = 'Must have a source grid to plot a snapshot.'
            raise ValueError(msg)
        
        # ToDo: Replace all the hardcoded geographical boundary values!
        map_x = self.sourcegrid[0][0::resolution]
        map_y = self.sourcegrid[1][0::resolution]
                                 
        plot.plot_grid(map_x,map_y,self.get_snapshot(t,resolution=resolution),**kwargs)
    
    def update_stats(self):
        
        if self.w != 'r':
            print('Updating stats...')
            self.file['stats'].attrs['ntraces'] = len(self.data[:,0]) if not self.complex else\
            len(self.data_r[:,0])
            self.file['stats'].attrs['nt'] = len(self.data[0,:]) if not self.complex else\
            len(self.data_r[0,:])
            self.file['stats'].attrs['complex'] = self.complex
            
            if 'stats' not in self.file.keys():
                self.file.create_dataset('stats',data=(0,))
            for (key,value) in self.stats.items():
                self.file['stats'].attrs[key] = value
            
            #print(self.file['stats'])
            #self.file.flush()
    
    #def write_sourcegrid(self):
     #   self.file.create_dataset('sourcegrid',data=self.sourcegrid)
     #   self.file.flush()
    

    def __enter__(self):
        return self
    
    def __exit__(self,type,value,traceback):
       
        self.update_stats()
        
        #ToDo update the stats
        
        self.file.close()
        
        
        
    
