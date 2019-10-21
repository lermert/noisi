import numpy as np
from math import pi, sin, cos, sqrt
from obspy.geodetics import gps2dist_azimuth
try:
    import cartopy.io.shapereader as shpreader
    import shapely.geometry as sgeom
    from shapely.ops import unary_union
    from shapely.prepared import prep
except ImportError:
    pass
from warnings import warn


def geographical_distances(grid, location):

    def f(lat, lon, location):

        return abs(gps2dist_azimuth(lat, lon, location[0], location[1])[0])

    dist = np.array([f(lat, lon, location) for lat, lon in
                     zip(grid[1], grid[0])])
    return dist


def is_land(x, y, res="110m"):

    if 'prep' not in globals():
        raise ImportError("cartopy is needed to design ocean-only source.")
    assert(res in ["10m", "50m", "110m"]), "Resolution must be 10m, 50m, 110 m"

    land_shp_fname = shpreader.natural_earth(resolution=res,
                                             category='physical',
                                             name='land')

    land_geom = unary_union(list(shpreader.Reader(land_shp_fname).
                                 geometries()))
    land = prep(land_geom)
    is_land = np.zeros(len(x))
    for i in range(len(x)):
        is_land[i] = land.contains(sgeom.Point(x[i], y[i]))
    return is_land


def wgs84():

    # semi-major axis, in m
    a = 6378137.0

    # semi-minor axis, in m
    b = 6356752.314245

    # inverse flattening f
    f = a / (a - b)

    # squared eccentricity e
    e_2 = (a ** 2 - b ** 2) / a ** 2

    return(a, b, e_2, f)


def geograph_to_geocent(theta):
    # geographic to geocentric
    # https://en.wikipedia.org/wiki/Latitude#Geocentric_latitude
    e2 = wgs84()[2]
    theta = np.rad2deg(np.arctan((1 - e2) * np.tan(np.deg2rad(theta))))
    return theta


def geocent_to_geograph(theta):
    # the other way around
    e2 = wgs84()[2]
    theta = np.rad2deg(np.arctan(np.tan(np.deg2rad(theta)) / (1 - e2)))
    return(theta)


def len_deg_lon(lat):
    (a, b, e_2, f) = wgs84()

    # This is the length of one degree of longitude
    # approx. after WGS84, at latitude lat
    # in m
    lat = pi / 180 * lat
    dlon = (pi * a * cos(lat)) / 180 * sqrt((1 - e_2 * sin(lat) ** 2))
    return round(dlon, 5)


def len_deg_lat(lat):
    # This is the length of one degree of latitude
    # approx. after WGS84, between lat-0.5deg and lat+0.5 deg
    # in m
    lat = pi / 180 * lat
    dlat = 111132.954 - 559.822 * cos(2 * lat) + 1.175 * cos(4 * lat)
    return round(dlat, 5)


def get_spherical_surface_elements(lon, lat, r=6.378100e6):

    if len(lon) < 3:
        raise ValueError('Grid must have at least 3 points.')
    if len(lon) != len(lat):
        raise ValueError('Grid x and y must have same length.')

    # surfel
    surfel = np.zeros(lon.shape)
    colat = 90. - lat

    # find neighbours
    for i in range(len(lon)):

        # finding the relevant neighbours is very specific to how
        # the grid is set up here (in rings of constant colatitude)!
        # get the nearest longitude along the current colatitude
        current_colat = colat[i]
        if current_colat in [0., 180.]:
            # surface area will be 0 at poles.
            continue

        colat_idx = np.where(colat == current_colat)
        lon_idx_1 = np.argsort(np.abs(lon[colat_idx] - lon[i]))[1]
        lon_idx_2 = np.argsort(np.abs(lon[colat_idx] - lon[i]))[2]
        closest_lon_1 = lon[colat_idx][lon_idx_1]
        closest_lon_2 = lon[colat_idx][lon_idx_2]

        if closest_lon_1 > lon[i] and closest_lon_2 > lon[i]:
            d_lon = np.abs(min(closest_lon_2, closest_lon_1) - lon[i])

        elif closest_lon_1 < lon[i] and closest_lon_2 < lon[i]:
            d_lon = np.abs(max(closest_lon_2, closest_lon_1) - lon[i])

        else:
            if closest_lon_1 != lon[i] and closest_lon_2 != lon[i]:
                d_lon = np.abs(closest_lon_2 - closest_lon_1) * 0.5
            else:
                d_lon = np.max(np.abs(closest_lon_2 - lon[i]),
                               np.abs(closest_lon_1 - lon[i]))

        colats = np.array(list(set(colat.copy())))
        colat_idx_1 = np.argsort(np.abs(colats - current_colat))[1]
        closest_colat_1 = colats[colat_idx_1]
        colat_idx_2 = np.argsort(np.abs(colats - current_colat))[2]
        closest_colat_2 = colats[colat_idx_2]

        if (closest_colat_2 > current_colat
            and closest_colat_1 > current_colat):

            d_colat = np.abs(min(closest_colat_1,
                                 closest_colat_2) - current_colat)

        elif (closest_colat_2 < current_colat and
              closest_colat_1 < current_colat):
            d_colat = np.abs(max(closest_colat_1,
                                 closest_colat_2) - current_colat)

        else:
            if (closest_colat_2 != current_colat
                and closest_colat_1 != current_colat):
                d_colat = 0.5 * np.abs(closest_colat_2 - closest_colat_1)
            else:
                d_colat = np.max(np.abs(closest_colat_2 - current_colat),
                                 np.abs(closest_colat_1 - current_colat))

        surfel[i] = (np.deg2rad(d_lon) *
                     np.deg2rad(d_colat) *
                     sin(np.deg2rad(colat[i])) * r ** 2)

    return(surfel)


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
    assert xmax <= 180., 'Longitude must be within -180 -- 180 degrees.'
    assert xmin >= -180., 'Longitude must be within -180 -- 180 degrees.'

    gridx = []
    gridy = []

    if ymin == -90.:
        ymin = -89.999
        warn("Resetting lat_min to -89.999 degree")
    if ymax == 90.:
        ymax = 89.999
        warn("Resetting lat_max to 89.999 degree")
    lat = ymin
    # do not start or end at pole because 1 deg of longitude is 0 m there
    while lat <= ymax:
        d_lon = dx / len_deg_lon(lat)
        # the starting point of each longitudinal circle is randomized
        perturb = np.random.rand(1)[0] * d_lon - 0.5 * d_lon
        lon = min(max(xmin + perturb, -180.), 180.)

        while lon <= xmax:
            gridx.append(lon)
            gridy.append(lat)
            lon += d_lon
        d_lat = dx / len_deg_lat(lat)
        lat += d_lat
    return list((gridx, gridy))
