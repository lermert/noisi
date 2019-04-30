import numpy as np
import yaml
import os
import io
from obspy.geodetics import gps2dist_azimuth
from noisi_v1.util.geo import len_deg_lat, len_deg_lon
from warnings import warn


def points_on_ell(dx, xmin=-180., xmax=180., ymin=-89.999, ymax=89.999):
    """
    Calculate an approximately equally spaced grid on an
    elliptical Earth's surface.
    :type dx: float
    :param dx: spacing in latitudinal and longitudinal direction in meter
    :returns: np.array(latitude, longitude) of grid points,
    where -180<=lon<180     and -90 <= lat < 90
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
    if ymax == 90.:
        ymax = 89.999
        warn("Resetting lat_max to 89.999 degree")
        # do not start or end at pole because 1 deg of longitude is 0 m there
    while lat <= ymax:
        
        d_lon = dx / len_deg_lon(lat)
        # the starting point of each longitudinal circle is randomized
        lon = xmin + np.random.rand(1)[0] * d_lon

        while lon <= xmax:
            gridx.append(lon)
            gridy.append(lat)
            d_lon = dx / len_deg_lon(lat)
            lon += d_lon
        d_lat = dx / len_deg_lat(lat)
        lat += d_lat

    return list((gridx, gridy))


def create_sourcegrid(config):

    print(config)
    grid = points_on_ell(config['grid_dx'],
                         xmin=config['grid_lon_min'],
                         xmax=config['grid_lon_max'],
                         ymin=config['grid_lat_min'],
                         ymax=config['grid_lat_max'])
    sources = np.zeros((2, len(grid[0])))
    sources[0:2, :] = grid

    print('Number of gridpoints: ', np.size(grid) / 2)

    return sources


def setup_sourcegrid(args):
    configfile = os.path.join(args.project_path, 'config.yml')
    with io.open(configfile, 'r') as fh:
        config = yaml.safe_load(fh)

    grid_filename = os.path.join(config['project_path'], 'sourcegrid.npy')
    sourcegrid = create_sourcegrid(config)

    # write to .npy
    np.save(grid_filename, sourcegrid)

    return()
