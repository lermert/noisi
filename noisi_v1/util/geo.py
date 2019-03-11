import numpy as np
from math import pi, sin, cos, sqrt
from obspy.geodetics import gps2dist_azimuth


def wgs84():

    # semi-major axis, in m
    a = 6378137.0

    # semi-minor axis, in m
    b = 6356752.314245

    # inverse flattening f
    f = a/(a-b)

    # squared eccentricity e
    e_2 = (a**2-b**2)/a**2
    
    return(a,b,e_2,f)

        # geographic to geocentric
def geograph_to_geocent(theta):
    # https://en.wikipedia.org/wiki/Latitude#Geocentric_latitude
    e2 = wgs84()[2]
    theta = np.rad2deg(np.arctan((1 - e2) * np.tan(np.deg2rad(theta))))
    return theta

def len_deg_lon(lat):
    
    (a,b,e_2,f) = wgs84()
    # This is the length of one degree of longitude 
    # approx. after WGS84, at latitude lat
    # in m
    lat = pi/180*lat
    dlon = (pi*a*cos(lat))/180*sqrt((1-e_2*sin(lat)**2))
    return round(dlon,5)

def len_deg_lat(lat):
    # This is the length of one degree of latitude 
    # approx. after WGS84, between lat-0.5deg and lat+0.5 deg
    # in m
    lat = pi/180*lat
    dlat = 111132.954 - 559.822 * cos(2*lat) + 1.175*cos(4*lat)
    return round(dlat,5)

def get_spherical_surface_elements(lon,lat):

    # radius...assuming spherical Earth here
    r = 6.378100e6
    # surfel
    surfel = np.zeros(lon.shape)
    colat = 90. - lat

    # find neighbours
    for i in range(len(lon)):

        # finding the relevant neighbours is very specific to how the grid is
        # set up here (in rings of constant colatitude)!

        # get the nearest longitude along the current colatitude 
        current_colat = colat[i]
        if current_colat in [0.,180.]:
            # surface area will be 0 at poles.
            continue

        colat_idx = np.where(colat==current_colat)
        lon_idx_1 = np.argsort(np.abs(lon[colat_idx]-lon[i]))[1]
        lon_idx_2 = np.argsort(np.abs(lon[colat_idx]-lon[i]))[2]
        closest_lon_1 = lon[colat_idx][lon_idx_1]
        closest_lon_2 = lon[colat_idx][lon_idx_2]
        
        if closest_lon_1 > lon[i] and closest_lon_2 > lon[i]:
            d_lon = np.abs(min(closest_lon_2,closest_lon_1)-lon[i])

        elif closest_lon_1 < lon[i] and closest_lon_2 < lon[i]:
            d_lon = np.abs(max(closest_lon_2,closest_lon_1)-lon[i])
            
        else:
            if closest_lon_1 != lon[i] and closest_lon_2 != lon[i]:
                d_lon = np.abs(closest_lon_2 - closest_lon_1) * 0.5
            else:
                d_lon = np.max(np.abs(closest_lon_2-lon[i]),
                               np.abs(closest_lon_1-lon[i]))

    # wuah...I am fed up so let's do this in a slightly rubbish manner
        colats = np.array(list(set(colat.copy())))
        colat_idx_1 = np.argsort(np.abs(colats-current_colat))[1]
        closest_colat_1 = colats[colat_idx_1]
        colat_idx_2 = np.argsort(np.abs(colats-current_colat))[2]
        closest_colat_2 = colats[colat_idx_2]
        

        if (closest_colat_2 > current_colat and 
            closest_colat_1 > current_colat):
            d_colat = np.abs(min(closest_colat_1,
                    closest_colat_2)-current_colat)
            
        elif (closest_colat_2 < current_colat and 
            closest_colat_1 < current_colat):
            d_colat = np.abs(max(closest_colat_1,
                closest_colat_2)-current_colat)
            
        else:
            if (closest_colat_2 != current_colat 
                and closest_colat_1 != current_colat):
                d_colat = 0.5 * np.abs(closest_colat_2-closest_colat_1)
            else:
                d_colat = np.max(np.abs(closest_colat_2-current_colat),
                               np.abs(closest_colat_1-current_colat))

        surfel[i] = np.deg2rad(d_lon) *\
        np.deg2rad(d_colat) * sin(np.deg2rad(colat[i])) * r**2


    return(surfel)





#ToDo: Tests
def points_on_sphere(dx,xmin=-180.,xmax=180.,ymin=-90.,ymax=90.,c_centr=None,\
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
    
    while lat <= ymax:
        d_lat = dx / len_deg_lat(lat)
        lon = xmin
        while lon <= xmax:
            
            if c_centr and radius:
                if gps2dist_azimuth(lat,lon,c_centr[0],c_centr[1])[0] > radius:
                    if abs(lat) != 90.:
                        d_lon = dx / len_deg_lon(lat)
                        lon += d_lon
                        continue
                    else:
                        break
                    
            gridx.append(lon)
            gridy.append(lat)
            
            if abs(lat) == 90:
                # length of a degree longitude will be 0.
                break
            else:
                d_lon = dx / len_deg_lon(lat)
                lon += d_lon
        lat += d_lat # do not start at pole or zero division will raise...
        
            
    # return values sorted by longitude, because basemap complains otherwise.
    grid = list(zip(*sorted(zip(gridx, gridy), key=lambda it: it[0])))
    return grid
    
