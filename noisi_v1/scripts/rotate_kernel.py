import numpy as np
import h5py
from math import radians, cos, sin
from noisi_v1 import WaveField
from pandas import read_csv
from obspy.geodetics import gps2dist_azimuth
from obspy import read
from noisi_v1.scripts.correlation import config_params
from noisi_v1.scripts.kernel import open_adjoint_sources


def rotation_matrix(baz_in_degrees):
    baz_rad = radians(baz_in_degrees)
    rot_er = -sin(baz_rad)
    rot_nr = -cos(baz_rad)
    rot_et = -cos(baz_rad)
    rot_nt = sin(baz_rad)
    return(np.asarray([[1, 0, 0], [0, rot_nr, rot_nt], [0, rot_er, rot_et]]))

def rotate_kernel(input_temp_wfs, adjoint_sources, sta1, sta2, stationlist):
    # input_temp_files: list of WaveField objects of Green function corrs
    # adjoint sources: list of adjoint sources as obspy trace objects,
    # in the order: list of components[list of filter bands[list of causal, acausal[]]]

    # find back azimuth
    stationlist = read_csv(stationlist)
    lat1 = stationlist[stationlist.sta == sta1]["lat"].values[0]
    lon1 = stationlist[stationlist.sta == sta1]["lon"].values[0]
    lat2 = stationlist[stationlist.sta == sta2]["lat"].values[0]
    lon2 = stationlist[stationlist.sta == sta2]["lon"].values[0]
    print(lat1, lon1, lat2, lon2)
    baz_in_degrees = gps2dist_azimuth(lat1, lon1, lat2, lon2)[-1]

    # get rotation matrix
    M = rotation_matrix(baz_in_degrees)

    # allocate kernel array
    filtcnt = len(adjoint_sources[0])
    kern = np.zeros((filtcnt, input_temp_wfs[0].stats["ntraces"],
                     len(adjoint_sources[0][0])))
    print(kern.shape)
    # loop over source locs
    # compile the correlation matrix
    # rotate
    # apply adjoint sources
    # save


wf = WaveField("iteration_0/kern/G.SSB..MXR--MN.BNI..MXT.h5_temp")
wf2 = WaveField("iteration_0/kern/G.SSB..MXR--MN.BNI..MXR.h5_temp")


class args(object):
    """docstring for args"""
    def __init__(self):
        args.source_model = "/home/lermert/Dropbox/example/example/source_1"
        args.step = 0
        args.ignore_network = 1
        args.steplengthrun = 0


callargs = args()
all_conf = config_params(args, 1, 1, 1)
adjt = ["iteration_0/adjt/G.SSB..MXR--MN.BNI..MXT"]
n_corr = read(adjt[0] + "*.0.sac")[0].stats.npts
adjoint_sources = open_adjoint_sources(all_conf, adjt, n_corr)
print(adjoint_sources)
rotate_kernel([wf, wf2], adjoint_sources, "SSB", "BNI",
              "/home/lermert/Dropbox/example/example/stationlist.csv")
