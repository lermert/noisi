import numpy as np
import yaml
import os
import io
from noisi_v1.util.geo import points_on_ell
from noisi_v1.util.plot import plot_sourcegrid
import cartopy.crs as ccrs


def create_sourcegrid(config):

    if config['verbose']:
        print(config)
    grid = points_on_ell(config['grid_dx'],
                         xmin=config['grid_lon_min'],
                         xmax=config['grid_lon_max'],
                         ymin=config['grid_lat_min'],
                         ymax=config['grid_lat_max'])
    sources = np.zeros((2, len(grid[0])))
    sources[0:2, :] = grid

    if config['verbose']:
        print('Number of gridpoints: ', np.size(grid) / 2)

    return sources


def setup_sourcegrid(args, comm, size, rank):
    configfile = os.path.join(args.project_path, 'config.yml')
    with io.open(configfile, 'r') as fh:
        config = yaml.safe_load(fh)

    grid_filename = os.path.join(config['project_path'], 'sourcegrid.npy')
    sourcegrid = create_sourcegrid(config)

    # plot
    plot_sourcegrid(sourcegrid, outfile=os.path.join(config['project_path'],
                                                     'sourcegrid.png'),
                    proj=ccrs.PlateCarree)

    # write to .npy
    np.save(grid_filename, sourcegrid)

    return()
