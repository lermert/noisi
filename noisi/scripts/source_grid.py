import numpy as np
import json
import os
import io
from obspy.geodetics import gps2dist_azimuth
from noisi.util.geo import len_deg_lat, len_deg_lon
from warnings import warn
    

def points_on_sphere(dx,xmin=-180.,xmax=180.,ymin=-89.999,ymax=89.999,c_centr=None,\
radius=None):
    """
    Calculate a more or less equally spaced grid on spherical Earth's surface.
    :param dx: spacing in latitudinal and longitudinal direction in meter
    :type c_centr: Tuple
    :param c_centr: Specify a central location
    :type radius: float
    :param radius: Radius around central location in m; no sources beyond this will be included
    :returns: np.array(latitude, longitude) of grid points, where -180<=lon<180     and -90 <= lat < 90
    """
    
    if xmax <= xmin or ymax <= ymin:
        msg = 'Lower bounds must be lower than upper bounds.'
        raise ValueError(msg)

    
    gridx = []
    gridy = []
    
    lat = ymin
    if ymin == -90.:
        ymin = -89.999
        warn("Resetting lat_min to -89.999 degree")
    
    while lat <= ymax:
        d_lat = dx / len_deg_lat(lat)
        d_lon = dx / len_deg_lon(lat)
            
        lon = xmin + np.random.rand(1)[0] * d_lon

        while lon <= xmax:
                    
            gridx.append(lon)
            gridy.append(lat)

            if c_centr and radius:
                if gps2dist_azimuth(lat,lon,c_centr[0],c_centr[1])[0] > radius:
                    print(lat,lon,gps2dist_azimuth(lat,lon,c_centr[0],c_centr[1])[0])
                    if abs(lat) != 90.:
                        d_lon = dx / len_deg_lon(lat)
                        lon += d_lon
                        continue
                    else:
                        break

            
            if abs(lat) == 90:
                # length of a degree longitude will be 0.
                break
            else:
                d_lon = dx / len_deg_lon(lat)
                lon += d_lon
        lat += d_lat # do not start at pole or zero division will raise...
        
            
    # return values sorted by longitude, basemap likes it.
    grid = list(zip(*sorted(zip(gridx, gridy), key=lambda it: it[0])))
    return list((gridx,gridy))#grid



    
def create_sourcegrid(config):
    
    cfile = open(config,'r')
    config = json.load(cfile)
    cfile.close()
    # ToDo: Pretty printing of all dictionaries such as this one.
    print(config)
    
    #ToDo read extra parameters into configuration
    grid = points_on_sphere(config['grid_dx'],
    xmin=config['grid_lon_min'],
    xmax=config['grid_lon_max'],
    ymin=config['grid_lat_min'],
    ymax=config['grid_lat_max'],
    c_centr=config['grid_coord_centr'],radius=config['grid_radius'])
   
    sources = np.zeros((2,len(grid[0])))
    #sources[0,:] = ids
    sources[0:2,:] = grid
    
    print('Number of gridpoints:',np.size(grid)/2)
    
    return sources
    
    
#def grid_to_specfem_stations(grid,outfile):
#    """
#    Write noisesource grid to disk as specfem compatible station list.
#    """
#    
#    fid = open(outfile,'w')
#    for i in range(len(grid[0,:])):
#        fid.write('%08g SRC %10.8f  %10.8f 0.0 0.0\n'\
#            %(i,grid[1,i],grid[0,i]))
#    
#    fid.close()
#    

def setup_sourcegrid(configfile,out='specfem'):
    
    sourcegrid = create_sourcegrid(configfile)
    
    with io.open(configfile,'r') as fh:
        config = json.load(fh)
    grid_filename = os.path.join(config['project_path'],'sourcegrid.npy')
    
    
    
    # write to .npy
    np.save(grid_filename,sourcegrid)
    # write to specfem friendly text file
    # or whatever
    #if out == 'specfem':
        #stations_filename = os.path.join(config['project_path'],'STATIONS')
        #grid_to_specfem_stations(sourcegrid,stations_filename)
    
