"""
Smoothing routine for noisi
:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np
from math import sqrt, pi, cos, sin
import sys
import math
from numba import jit, int32, float64, boolean
from glob import glob
import os
"""
Lines 25 - 215:
Various geodetic utilities for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563

@jit(float64(float64, float64), nopython=True)
def _isclose(a, b):
    """
    Equivalent of the :meth:`math.isclose` method compatible with python 2.7.
    """
    rel_tol=1e-09
    abs_tol=0.0
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

@jit("void(float64)", nopython=True)
def _check_latitude(latitude):
    """
    Check whether latitude is in the -90 to +90 range.
    """
    if latitude is None:
        return
    if latitude > 90 or latitude < -90:
        msg = 'Latitude out of bounds! (-90 <= Lat. <=90)'
        raise ValueError(msg)

@jit(float64(float64), nopython=True)
def _normalize_longitude(longitude):
    """
    Normalize longitude in the -180 to +180 range.
    """
    while longitude > 180:
        longitude -= 360
    while longitude < -180:
        longitude += 360
    return longitude

# # Vincenty inversion (python calculation) from obspy
@jit(float64(float64, float64, float64, float64), nopython=True)
def calc_vincenty_inverse(lat1, lon1, lat2, lon2):
    """
    Vincenty Inverse Solution of Geodesics on the Ellipsoid.

    Computes the distance between two geographic points on the WGS84
    ellipsoid and the forward and backward azimuths between these points.

    :param lat1: Latitude of point A in degrees (positive for northern,
        negative for southern hemisphere)
    :param lon1: Longitude of point A in degrees (positive for eastern,
        negative for western hemisphere)
    :param lat2: Latitude of point B in degrees (positive for northern,
        negative for southern hemisphere)
    :param lon2: Longitude of point B in degrees (positive for eastern,
        negative for western hemisphere)
    :param a: Radius of Earth in m. Uses the value for WGS84 by default.
    :param f: Flattening of Earth. Uses the value for WGS84 by default.
    :return: (Great circle distance in m, azimuth A->B in degrees,
        azimuth B->A in degrees)
    :raises: This method may have no solution between two nearly antipodal
        points; an iteration limit traps this case and a ``StopIteration``
        exception will be raised.

    .. note::
        This code is based on an implementation incorporated in
        Matplotlib Basemap Toolkit 0.9.5 

        Algorithm from Geocentric Datum of Australia Technical Manual.

        * http://www.icsm.gov.au/gda/
        * http://www.icsm.gov.au/gda/gdatm/gdav2.3.pdf, pp. 15

        It states::

            Computations on the Ellipsoid

            There are a number of formulae that are available to calculate
            accurate geodetic positions, azimuths and distances on the
            ellipsoid.

            Vincenty's formulae (Vincenty, 1975) may be used for lines ranging
            from a few cm to nearly 20,000 km, with millimetre accuracy. The
            formulae have been extensively tested for the Australian region, by
            comparison with results from other formulae (Rainsford, 1955 &
            Sodano, 1965).

            * Inverse problem: azimuth and distance from known latitudes and
              longitudes
            * Direct problem: Latitude and longitude from known position,
              azimuth and distance.
    """
    # Check inputs
    _check_latitude(lat1)
    lon1 = _normalize_longitude(lon1)
    _check_latitude(lat2)
    lon2 = _normalize_longitude(lon2)
    a = 6378137.0
    f = 1 / 298.257223563

    b = a * (1 - f)  # semiminor axis

    if _isclose(lat1, lat2) and _isclose(lon1, lon2):
        dist = 0.
        alpha21 = 0.
        alpha12 = 0.

    else:
        # convert latitudes and longitudes to radians:
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        tan_u1 = (1 - f) * math.tan(lat1)
        tan_u2 = (1 - f) * math.tan(lat2)

        u_1 = math.atan(tan_u1)
        u_2 = math.atan(tan_u2)

        dlon = lon2 - lon1
        last_dlon = -4000000.0  # an impossible value
        omega = dlon

        # Iterate until no significant change in dlon or iterlimit has been
        # reached (http://www.movable-type.co.uk/scripts/latlong-vincenty.html)
        iterlimit = 100
        while (last_dlon < -3000000.0 or dlon != 0 and
               abs((last_dlon - dlon) / dlon) > 1.0e-9):
            sqr_sin_sigma = pow(math.cos(u_2) * math.sin(dlon), 2) + \
                pow((math.cos(u_1) * math.sin(u_2) - math.sin(u_1) *
                     math.cos(u_2) * math.cos(dlon)), 2)
            sin_sigma = math.sqrt(sqr_sin_sigma)

            cos_sigma = math.sin(u_1) * math.sin(u_2) + math.cos(u_1) * \
                math.cos(u_2) * math.cos(dlon)
            sigma = math.atan2(sin_sigma, cos_sigma)
            sin_alpha = math.cos(u_1) * math.cos(u_2) * math.sin(dlon) / \
                sin_sigma

            sqr_cos_alpha = 1 - sin_alpha * sin_alpha
            if _isclose(sqr_cos_alpha, 0):
                # Equatorial line
                cos2sigma_m = 0
            else:
                cos2sigma_m = cos_sigma - \
                    (2 * math.sin(u_1) * math.sin(u_2) / sqr_cos_alpha)

            c = (f / 16) * sqr_cos_alpha * (4 + f * (4 - 3 * sqr_cos_alpha))
            last_dlon = dlon
            dlon = omega + (1 - c) * f * sin_alpha * \
                (sigma + c * sin_sigma *
                    (cos2sigma_m + c * cos_sigma *
                        (-1 + 2 * pow(cos2sigma_m, 2))))

            iterlimit -= 1
            if iterlimit < 0:
                # iteration limit reached
                dist = np.nan
                alpha21 = np.nan
                alpha12 = np.nan


        u2 = sqr_cos_alpha * (a * a - b * b) / (b * b)
        _a = 1 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
        _b = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
        delta_sigma = _b * sin_sigma * \
            (cos2sigma_m + (_b / 4) *
                (cos_sigma * (-1 + 2 * pow(cos2sigma_m, 2)) - (_b / 6) *
                    cos2sigma_m * (-3 + 4 * sqr_sin_sigma) *
                    (-3 + 4 * pow(cos2sigma_m, 2))))

        dist = b * _a * (sigma - delta_sigma)
        alpha12 = math.atan2(
            (math.cos(u_2) * math.sin(dlon)),
            (math.cos(u_1) * math.sin(u_2) -
                math.sin(u_1) * math.cos(u_2) * math.cos(dlon)))
        alpha21 = math.atan2(
            (math.cos(u_1) * math.sin(dlon)),
            (-math.sin(u_1) * math.cos(u_2) +
                math.cos(u_1) * math.sin(u_2) * math.cos(dlon)))

        if alpha12 < 0.0:
            alpha12 = alpha12 + (2.0 * math.pi)
        if alpha12 > (2.0 * math.pi):
            alpha12 = alpha12 - (2.0 * math.pi)

        alpha21 = alpha21 + math.pi

        if alpha21 < 0.0:
            alpha21 = alpha21 + (2.0 * math.pi)
        if alpha21 > (2.0 * math.pi):
            alpha21 = alpha21 - (2.0 * math.pi)

        # convert to degrees:
        alpha12 = alpha12 * 360 / (2.0 * math.pi)
        alpha21 = alpha21 * 360 / (2.0 * math.pi)

    return dist



# chord length as great circle distance on the spherical earth
@jit(float64[:](float64, float64, float64[:], float64[:]), nopython=True)
def chord_length(lat1, lon1, lat2, lon2):
    lat1 = np.pi * lat1 / 180.
    lon1 = np.pi * lon1 / 180.

    lat2 = np.pi * lat2 / 180.
    lon2 = np.pi * lon2 / 180.

    dx = np.cos(lat2) * np.cos(lon2) - np.cos(lat1) * np.cos(lon1)
    dy = np.cos(lat2) * np.sin(lon2) - np.cos(lat1) * np.sin(lon1)
    dz = np.sin(lat2) - np.sin(lat1)

    d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    d = d * 6371000.0
    return(d)


@jit(float64[:, :, :](float64[:, :, :], float64[:, :], int32, int32, float64, boolean), nopython=True)
def smooth_gaussian(values, coords, rank, size, sigma, ellipsoid):
    # coords format: (lon,lat)

    v_smooth = np.zeros(values.shape)
    print(v_smooth.shape)

    a = 1. / (sigma * sqrt(2. * pi))

    for j in range(rank, values.shape[1], size):
        dist = np.zeros(coords.shape[-1])
        if ellipsoid:
            for m in range(values.shape[1]):
                dist[m] = calc_vincenty_inverse(coords[1, j], coords[0, j], 
                                                coords[1, m], coords[0, m])
        else:
            dist = chord_length(coords[1, j], coords[0, j], 
                                       coords[1], coords[0])
        print(dist[1:10], sigma)
        weight = a * np.exp(-(dist) ** 2 / (2 * sigma ** 2))

        for i in range(values.shape[0]):
            for k in range(values.shape[-1]):
                if j % 100 == 0:
                    print(rank, i, j, k)
                v_smooth[i, j, k] = np.sum(np.multiply(weight, values[i, j, k])) / len(values[j])

    
    return(v_smooth)


def apply_smoothing_sphere(rank, size, values, coords, sigma, cap,
                           threshold, comm,
                           ellipsoid=True):

    sigma = float(sigma)
    cap = float(cap)
    threshold = float(threshold)

    # clip
    for i in range(values.shape[0]):
        for k in range(values.shape[-1]):
            perc_up = np.percentile(values[i, :, k], cap, overwrite_input=False)
            perc_dw = np.percentile(values[i, :, k], 100 - cap, overwrite_input=False)
            values[i, :, k] = np.clip(values[i, :, k], perc_dw, perc_up)
    # get the smoothed map; could use other functions than Gaussian here
    v_smooth = smooth_gaussian(values, coords, rank, size, sigma,
                          ellipsoid=ellipsoid)
    np.save("temp_{}.npy".format(rank), v_smooth)

    comm.barrier()
    v_s = np.zeros(values.shape)
    # rank 0: save the values
    if rank == 0:
        for r in range(size):
            v_s += np.load("temp_{}.npy".format(r))

    return(v_s)


def smooth_new_new(inputfile, outputfile, coordfile, sigma, cap, thresh, comm, size,
           rank, ntraces, gradshape):

    for ixs in range(len(sigma)):
        sigma[ixs] = float(sigma[ixs])
    sigma = sigma[-1]

    if rank == 0:
        coords = np.load(coordfile)
    else:
        coords = np.zeros(ntraces)

    coords = comm.bcast(coords, root=0)

    if rank == 0:
        values = np.array(np.load(inputfile), ndmin=3)
    else:
        values = np.zeros(gradshape)

    values = comm.bcast(values, root=0)

    smoothed_values = np.zeros(values.shape)

    v = apply_smoothing_sphere(rank, size, values,
                               coords, sigma, cap, threshold=thresh,
                               comm=comm)
    comm.barrier()

    if rank == 0:
        smoothed_values = v
    
    if rank == 0:
        np.save(outputfile, smoothed_values / (smoothed_values.max() + np.finfo(smoothed_values.min()).eps))
        os.system("rm temp_?.npy")
        os.system("rm temp_??.npy")
        os.system("rm temp_???.npy")

    else:
        pass
    return()


