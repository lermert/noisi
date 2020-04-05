import numpy as np
import h5py
from math import radians, cos, sin
from noisi_v1 import WaveField
from pandas import read_csv
from obspy.geodetics import gps2dist_azimuth
from obspy import read
from noisi_v1 import NoiseSource
import os

def rotation_matrix(baz_in_degrees):
    baz_rad = radians(baz_in_degrees)
    rot_er = -sin(baz_rad)
    rot_nr = -cos(baz_rad)
    rot_et = -cos(baz_rad)
    rot_nt = sin(baz_rad)
    return(np.asarray([[1, 0, 0], [0, rot_nr, rot_nt], [0, rot_er, rot_et]]))

def assemble_rotated_kernel(temp_kern_files, output_files, adjt_srcs, stationlist):
    # input_temp_files: list of WaveField objects of Green function corrs
    # adjoint sources: list of adjoint sources as obspy trace objects,
    # in the order: list of components[list of filter bands[list of causal, acausal[]]]


    # find back azimuth
    stationlist = read_csv(stationlist)
    sta1 = os.path.basename(temp_kern_files[0]).split(".")[1]
    sta2 = os.path.basename(temp_kern_files[0]).split(".")[4]
    lat1 = stationlist[stationlist.sta == sta1]["lat"].values[0]
    lon1 = stationlist[stationlist.sta == sta1]["lon"].values[0]
    lat2 = stationlist[stationlist.sta == sta2]["lat"].values[0]
    lon2 = stationlist[stationlist.sta == sta2]["lon"].values[0]
    print(lat1, lon1, lat2, lon2)
    baz_in_degrees = gps2dist_azimuth(lat1, lon1, lat2, lon2)[-1]

    # get rotation matrix
    M = rotation_matrix(baz_in_degrees)

    # open the temp files
    f_ee = h5py.File(temp_kern_files[0], "r")
    f_en = h5py.File(temp_kern_files[1], "r")
    f_ez = h5py.File(temp_kern_files[2], "r")
    f_ne = h5py.File(temp_kern_files[3], "r")
    f_nn = h5py.File(temp_kern_files[4], "r")
    f_nz = h5py.File(temp_kern_files[5], "r")
    f_ze = h5py.File(temp_kern_files[6], "r")
    f_zn = h5py.File(temp_kern_files[7], "r")
    f_zz = h5py.File(temp_kern_files[8], "r")

    # allocate kernel array
    filtcnt = len(adjt_srcs[0])
    specs = [b for b in list(f_ee.keys()) if "data" in b]
    speccnt = len(specs) 
    ntraces = f_ee["stats"].attrs["ntraces"]
    nt = f_ee["data"].shape[-1]
    kern_tt = np.zeros((speccnt, filtcnt, ntraces,
                        len(adjt_srcs[0][0])))
    kern_rr = np.zeros((speccnt, filtcnt, ntraces,
                        len(adjt_srcs[0][0])))
    kern_zz = np.zeros((speccnt, filtcnt, ntraces,
                        len(adjt_srcs[0][0])))
    kern_zr = np.zeros((speccnt, filtcnt, ntraces,
                        len(adjt_srcs[0][0])))
    kern_zt = np.zeros((speccnt, filtcnt, ntraces,
                        len(adjt_srcs[0][0])))
    kern_rt = np.zeros((speccnt, filtcnt, ntraces,
                        len(adjt_srcs[0][0])))
    kern_rz = np.zeros((speccnt, filtcnt, ntraces,
                        len(adjt_srcs[0][0])))
    kern_tr = np.zeros((speccnt, filtcnt, ntraces,
                        len(adjt_srcs[0][0])))
    kern_tz = np.zeros((speccnt, filtcnt, ntraces,
                        len(adjt_srcs[0][0])))

    print(kern_tt.shape)
    # loop over source locs
    for i in range(ntraces):
        # compile the correlation matrix
        for ix_spec in range(speccnt):
            if i % 1000 == 1: print(i)
            K = np.zeros((nt, 3, 3))
            ix_s = specs[ix_spec]
            K[:, 0, 0] = f_zz[ix_s][i, :]
            K[:, 0, 1] = f_zn[ix_s][i, :]
            K[:, 0, 2] = f_ze[ix_s][i, :]
            K[:, 1, 0] = f_nz[ix_s][i, :]
            K[:, 1, 1] = f_nn[ix_s][i, :]
            K[:, 1, 2] = f_ne[ix_s][i, :]
            K[:, 2, 0] = f_ez[ix_s][i, :]
            K[:, 2, 1] = f_en[ix_s][i, :]
            K[:, 2, 2] = f_ee[ix_s][i, :]
            # rotate
            K_rot = np.matmul(np.matmul(M.T, K), M)
            # apply adjoint sources
            for ix_f in range(filtcnt):
                a_rr = adjt_srcs[4][ix_f]
                a_rt = adjt_srcs[5][ix_f]
                a_rz = adjt_srcs[3][ix_f]
                a_tr = adjt_srcs[7][ix_f]
                a_tt = adjt_srcs[8][ix_f]
                a_tz = adjt_srcs[6][ix_f]
                a_zr = adjt_srcs[1][ix_f]
                a_zt = adjt_srcs[2][ix_f]
                a_zz = adjt_srcs[0][ix_f]
                
                if a_zz is None:
                    continue

                for j in range(len(a_zz)):
                    delta = a_zz[j].stats.delta
                    kern_zz[ix_spec, ix_f, i, j] = np.dot(K_rot[:, 0, 0],
                                                          a_zz[j].data) * delta
                    kern_zr[ix_spec, ix_f, i, j] = np.dot(K_rot[:, 0, 1],
                                                          a_zr[j].data) * delta
                    kern_zt[ix_spec, ix_f, i, j] = np.dot(K_rot[:, 0, 2],
                                                          a_zt[j].data) * delta
                    kern_rz[ix_spec, ix_f, i, j] = np.dot(K_rot[:, 1, 0],
                                                          a_rz[j].data) * delta   
                    kern_rr[ix_spec, ix_f, i, j] = np.dot(K_rot[:, 1, 1],
                                                          a_rr[j].data) * delta
                    kern_rt[ix_spec, ix_f, i, j] = np.dot(K_rot[:, 1, 2],
                                                          a_rt[j].data) * delta
                    kern_tz[ix_spec, ix_f, i, j] = np.dot(K_rot[:, 2, 0],
                                                          a_tz[j].data) * delta
                    kern_tr[ix_spec, ix_f, i, j] = np.dot(K_rot[:, 2, 1],
                                                          a_tr[j].data) * delta
                    kern_tt[ix_spec, ix_f, i, j] = np.dot(K_rot[:, 2, 2],
                                                          a_tt[j].data) * delta
    for ix_f in range(filtcnt):

        filename_zz = output_files[0] + '.{}.npy'.format(ix_f)
        np.save(filename_zz, kern_zz[:, ix_f, :, :])
        filename_zr = output_files[1] + '.{}.npy'.format(ix_f)
        np.save(filename_zr, kern_zr[:, ix_f, :, :])
        filename_zt = output_files[2] + '.{}.npy'.format(ix_f)
        np.save(filename_zt, kern_zt[:, ix_f, :, :])
        filename_rz = output_files[3] + '.{}.npy'.format(ix_f)
        np.save(filename_rz, kern_rz[:, ix_f, :, :])
        filename_rr = output_files[4] + '.{}.npy'.format(ix_f)
        np.save(filename_rr, kern_rr[:, ix_f, :, :])
        filename_rt = output_files[5] + '.{}.npy'.format(ix_f)
        np.save(filename_rt, kern_rt[:, ix_f, :, :])
        filename_tz = output_files[6] + '.{}.npy'.format(ix_f)
        np.save(filename_tz, kern_tz[:, ix_f, :, :])
        filename_tr = output_files[7] + '.{}.npy'.format(ix_f)
        np.save(filename_tr, kern_tr[:, ix_f, :, :])
        filename_tt = output_files[8] + '.{}.npy'.format(ix_f)
        np.save(filename_tt, kern_tt[:, ix_f, :, :])
