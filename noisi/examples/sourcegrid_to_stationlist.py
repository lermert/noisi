import numpy as np
import os

# - input --------------------------------------------------------------------
outdir = 'SEM_input'
gridfile = "example/sourcegrid.npy"
# - end input ----------------------------------------------------------------


def grid_to_stations_file(grid):
    """
    Write noisesource grid as STATIONS list that specfem and axisem3d can read
    """
    os.system("mkdir -p " + outdir)
    fid = open(os.path.join(outdir, 'STATIONS'), 'w')
    for i in range(len(grid[0, :])):
        fid.write('%08g SRC %10.8f  %10.8f 0.0 0.0\n'
                  % (i, grid[1, i], grid[0, i]))

    fid.close()


if __name__ == '__main__':

    grid = np.load(gridfile)
    grid_to_stations_file(grid)
