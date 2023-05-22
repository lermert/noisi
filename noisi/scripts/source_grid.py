import numpy as np
import yaml
import os
import io
from noisi.util.geo import points_on_ell, is_land
try:
    from noisi.util.plot import plot_sourcegrid
except ImportError:
    pass
try:
    import cartopy.crs as ccrs
except ImportError:
    pass
import pprint

def create_sourcegrid(config):

    if config['verbose']:
        print("Configuration used to set up source grid:", end="\n")
        pp = pprint.PrettyPrinter()
        pp.pprint(config)
    try:
        rando = config["random_shift_longitude"]
    except KeyError:
        rando = False
    grid = points_on_ell(config['grid_dx_in_m'],
                         xmin=config['grid_lon_min'],
                         xmax=config['grid_lon_max'],
                         ymin=config['grid_lat_min'],
                         ymax=config['grid_lat_max'],
                         randomize=rando)
    sources = np.zeros((2, len(grid[0])))
    sources[0:2, :] = grid

    if config['verbose']:
        print('Number of gridpoints: ', sources.shape[-1])

    return sources


def setup_sourcegrid(args, comm, size, rank):
    configfile = os.path.join(args.project_path, 'config.yml')
    with io.open(configfile, 'r') as fh:
        config = yaml.safe_load(fh)

    grid_filename = os.path.join(config['project_path'], 'sourcegrid.npy')
    sourcegrid = create_sourcegrid(config)

    if config["only_ocean_sources"]:
        land = is_land(sourcegrid[0], sourcegrid[1])
        print(land, land.shape)
        sourcegrid = sourcegrid[:, ~land]
    print("{} Ocean-only sources.".format(len(sourcegrid[0])))
    # plot
    try:
        plot_sourcegrid(sourcegrid,
                        outfile=os.path.join(config['project_path'],
                                             'sourcegrid.png'),
                        proj=ccrs.PlateCarree)
    except NameError:
        pass

    # write to .npy
    np.save(grid_filename, sourcegrid)

    return()
