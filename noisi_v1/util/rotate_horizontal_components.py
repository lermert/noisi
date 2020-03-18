from obspy.geodetics import gps2dist_azimuth
from math import sin, cos, radians
from obspy import read
from pandas import read_csv
from glob import glob
import os
import numpy as np


# rotate then correlate
def rotation_matrix(baz):
    baz_rad = radians(baz)
    rot_er = -sin(baz_rad)
    rot_nr = -cos(baz_rad)
    rot_et = -cos(baz_rad)
    rot_nt = sin(baz_rad)

    return(rot_nr, rot_nt, rot_er, rot_et)

def add_metadata(tr, seedid1, seedid2, lat1, lat2, lon1, lon2):

    tr.stats.network = seedid1.split('.')[0]
    tr.stats.station = seedid1.split('.')[1]
    tr.stats.location = ''
    tr.stats.channel = seedid1.split('.')[3]
    tr.stats.sac.stlo = lon1
    tr.stats.sac.stla = lat1
    tr.stats.sac.evlo = lon2
    tr.stats.sac.evla = lat2
    tr.stats.sac.kuser0 = seedid2.split('.')[0]
    tr.stats.sac.kevnm = seedid2.split('.')[1]
    tr.stats.sac.kuser1 = ''
    tr.stats.sac.kuser2 = seedid2.split('.')[3]
    tr.stats.sac.user0 = 1
    geoinf = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    tr.stats.sac.dist = geoinf[0]
    tr.stats.sac.az = geoinf[1]
    tr.stats.sac.baz = geoinf[2]


def apply_rotation(fls, stationlistfile, output_directory):
    # Apply rotation of the horizontal components for the cross-correlation output.
    # Note that the source input remains E, N, Z
    # i.e. instead of C_nn = sum(source_nn * crosscorr(G1_nn, G2_nn)) we will have
    # C_rr = sum(source_nn * crosscorr(G1_rn, G2_rn))

    # load data
    c_nn = read(fls[4])[0]
    c_ne = read(fls[3])[0]
    c_en = read(fls[1])[0]
    c_ee = read(fls[0])[0]
    c_ez = read(fls[2])[0]
    c_nz = read(fls[5])[0]
    c_ze = read(fls[6])[0]
    c_zn = read(fls[7])[0]

    # get locations
    meta = read_csv(stationlistfile)

    net1, sta1, loc1, cha1 = os.path.basename(fls[0]).split('.')[0: 4]
    try:
        net2, sta2, loc2, cha2 = os.path.basename(fls[0]).split('--')[1].split('.')[0: 4]
    except IndexError:
        net2, sta2, loc2, cha2 = os.path.basename(fls[0]).split('.')[4: 8]

    channel_basename = cha1[0: 2]

    print(sta1, sta2)
    lat1 = float(meta[meta['sta'] == sta1].iloc[0]['lat'])
    lat2 = float(meta[meta['sta'] == sta2].iloc[0]['lat'])
    lon1 = float(meta[meta['sta'] == sta1].iloc[0]['lon'])
    lon2 = float(meta[meta['sta'] == sta2].iloc[0]['lon'])

    baz = gps2dist_azimuth(lat1, lon1, lat2, lon2)[2]

    # get rotation matrix
    (m_nr, m_nt, m_er, m_et) = rotation_matrix(baz)
    
    # recombine
    c_rr_data = m_nr ** 2 * c_nn.data + m_er ** 2 * c_ee.data + (m_nr * m_er) * (c_ne.data + c_en.data)
    c_tt_data = m_nt ** 2 * c_nn.data + m_et ** 2 * c_ee.data + (m_nt * m_et) * (c_ne.data + c_en.data)
    c_rt_data = m_nr * m_nt * c_nn.data + m_nr * m_et * c_ne.data + m_er * m_nt * c_en.data + m_er * m_et * c_ee.data
    c_tr_data = m_nr * m_nt * c_nn.data + m_nt * m_er * c_ne.data + m_et * m_nr * c_en.data + m_er * m_et * c_ee.data
    
    c_zr_data = m_nr * c_zn.data + m_er * c_ze.data
    c_zt_data = m_nt * c_zn.data + m_et * c_ze.data
    c_rz_data = m_nr * c_nz.data + m_er * c_ez.data
    c_tz_data = m_nt * c_nz.data + m_et * c_ez.data


    tr_rr = c_nn.copy()
    tr_tt = c_ee.copy()
    tr_rt = c_ne.copy()
    tr_tr = c_en.copy()

    tr_zr = c_zn.copy()
    tr_zt = c_ze.copy()
    tr_rz = c_nz.copy()
    tr_tz = c_ez.copy()

    tr_rr.data = c_rr_data
    tr_tt.data = c_tt_data
    tr_rt.data = c_rt_data
    tr_tr.data = c_tr_data
    tr_zt.data = c_zt_data
    tr_zr.data = c_zr_data
    tr_tz.data = c_tz_data
    tr_rz.data = c_rz_data

    # copy / add metadata
    seedid1 = "{}.{}.{}.".format(net1, sta1, loc1)
    seedid2 = "{}.{}.{}.".format(net2, sta2, loc2)

    cha_r = channel_basename + "R"
    cha_t = channel_basename + "T"
    cha_z = channel_basename + "Z"
    add_metadata(tr_rr, seedid1 + cha_r, seedid2 + cha_r, lat1, lat2, lon1, lon2)
    add_metadata(tr_tt, seedid1 + cha_t, seedid2 + cha_t, lat1, lat2, lon1, lon2)
    add_metadata(tr_rt, seedid1 + cha_r, seedid2 + cha_t, lat1, lat2, lon1, lon2)
    add_metadata(tr_tr, seedid1 + cha_t, seedid2 + cha_r, lat1, lat2, lon1, lon2)
    add_metadata(tr_zr, seedid1 + cha_z, seedid2 + cha_r, lat1, lat2, lon1, lon2)
    add_metadata(tr_zt, seedid1 + cha_z, seedid2 + cha_t, lat1, lat2, lon1, lon2)
    add_metadata(tr_rz, seedid1 + cha_r, seedid2 + cha_z, lat1, lat2, lon1, lon2)
    add_metadata(tr_tz, seedid1 + cha_t, seedid2 + cha_z, lat1, lat2, lon1, lon2)
    

    
    # write
    filename_tr_rr = os.path.join(output_directory, seedid1 + cha_r + "--" + seedid2 + cha_r + ".sac")
    filename_tr_tt = os.path.join(output_directory, seedid1 + cha_t + "--" + seedid2 + cha_t + ".sac")
    filename_tr_rt = os.path.join(output_directory, seedid1 + cha_r + "--" + seedid2 + cha_t + ".sac")
    filename_tr_tr = os.path.join(output_directory, seedid1 + cha_t + "--" + seedid2 + cha_r + ".sac")
    filename_tr_rz = os.path.join(output_directory, seedid1 + cha_r + "--" + seedid2 + cha_z + ".sac")
    filename_tr_tz = os.path.join(output_directory, seedid1 + cha_t + "--" + seedid2 + cha_z + ".sac")
    filename_tr_zt = os.path.join(output_directory, seedid1 + cha_z + "--" + seedid2 + cha_t + ".sac")
    filename_tr_zr = os.path.join(output_directory, seedid1 + cha_z + "--" + seedid2 + cha_r + ".sac")
    
    
    
    
    tr_rr.write(filename_tr_rr, format="SAC")
    tr_tt.write(filename_tr_tt, format="SAC")
    tr_rt.write(filename_tr_rt, format="SAC")
    tr_tr.write(filename_tr_tr, format="SAC")
    tr_rz.write(filename_tr_rz, format="SAC")
    tr_tz.write(filename_tr_tz, format="SAC")
    tr_zt.write(filename_tr_zt, format="SAC")
    tr_zr.write(filename_tr_zr, format="SAC")
    