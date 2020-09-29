"""
Smoothing routine for noisi
:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np
from math import sqrt, pi
import sys

def get_distance(gridx, gridy, gridz, x, y, z):
    xd = gridx - x
    yd = gridy - y
    zd = gridz - z
    return np.sqrt(np.power(xd, 2) + np.power(yd, 2) + np.power(zd, 2))


def smooth_gaussian(values, coords, rank, size, sigma, r=6371000.,
                    threshold=1e-16):
    # coords format: (lon,lat)

    # step 1: Cartesian coordinates of map
    theta = np.deg2rad(-coords[1] + 90.)
    phi = np.deg2rad(coords[0] + 180.)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    v_smooth = np.zeros(values.shape)

    a = 1. / (sigma * sqrt(2. * pi))
    for i in range(rank, len(values), size):
        xp, yp, zp = (x[i], y[i], z[i])
        dist = get_distance(x, y, z, xp, yp, zp)
        weight = a * np.exp(-(dist) ** 2 / (2 * sigma ** 2))
        idx = weight >= threshold
        v_smooth[i] = np.sum(np.multiply(weight[idx], values[idx])) / idx.sum()

    return v_smooth


def apply_smoothing_sphere(rank, size, values, coords, sigma, cap,
                           threshold, comm):

    sigma = float(sigma)
    cap = float(cap)
    threshold = float(threshold)

    # clip
    perc_up = np.percentile(values, cap, overwrite_input=False)
    perc_dw = np.percentile(values, 100 - cap, overwrite_input=False)
    values = np.clip(values, perc_dw, perc_up)

    # get the smoothed map; could use other functions than Gaussian here
    v_s = smooth_gaussian(values, coords, rank, size, sigma,
                          threshold=threshold)

    comm.barrier()

    # collect the values
    v_s_all = comm.gather(v_s, root=0)
    # rank 0: save the values
    if rank == 0:
        v_s = np.zeros(v_s.shape)
        for i in range(size):
            v_s += v_s_all[i]

        return(v_s)


def smooth(inputfile, outputfile, coordfile, sigma, cap, thresh, comm, size,
           rank):

    for ixs in range(len(sigma)):
        sigma[ixs] = float(sigma[ixs])

    coords = np.load(coordfile)
    values = np.array(np.load(inputfile), ndmin=3)
    # if values.shape[0] > values.shape[-1]:
    #     values = np.transpose(values)
    # shape of kernel:
    # n_components, n_locations, n_spectra
    smoothed_values = np.zeros(values.shape)
    for i in range(values.shape[0]):
        for k in range(values.shape[-1]):
            array_in = values[i, :, k]
            try:
                sig = sigma[i]
            except IndexError:
                sig = sigma[-1]

            v = apply_smoothing_sphere(rank, size, array_in,
                                       coords, sig, cap, threshold=thresh,
                                       comm=comm)
            comm.barrier()

            if rank == 0:
                smoothed_values[i, :, k] = v

        comm.barrier()

    if outputfile is not None:
        if rank == 0:
            np.save(outputfile, smoothed_values / (smoothed_values.max() + np.finfo(smoothed_values.min()).eps))
            return()
        else:
            return()
    else:
            return(smoothed_values / (smoothed_values.max() + np.finfo(smoothed_values.min()).eps))


if __name__ == '__main__':

    # pass in: input_file, output_file, coord_file, sigma
    # open the files
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    coordfile = sys.argv[3]
    sigma = sys.argv[4].split(',')
    cap = float(sys.argv[5])
    try:
        thresh = float(sys.argv[6])
    except IndexError:
        thresh = 1.e-12

    smooth(inputfile, outputfile, coordfile, sigma, cap, thresh)
