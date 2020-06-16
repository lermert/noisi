import numpy as np
from pandas import read_csv
import os
from noisi.util.geo import geocent_to_geograph

# - input --------------------------------------------------------------------
hdur_pointsource = 0.0
outdir = "SEM_input"
stationlist = "example/stationlist.csv"
solver = "axisem3d"
use_buried = False
use_topo = False
components = ["Z"] #  source force component
factor_force = 1.e9
# - end input ----------------------------------------------------------------


def stations_to_forcesolutions(stationlist, hdur, outdir, solver,
                               channel='MXZ', use_topo=False, use_buried=False,
                               source_type='Gauss', factor_force=1.e10,
                               station_ref="geographic"):

    stationlist = read_csv(stationlist)

    for i in range(len(stationlist)):
        station = stationlist.iloc[i].sta
        network = stationlist.iloc[i].net
        if station_ref == "geocentric":  # mainly for Japanese geogr datum
            latitude = geocent_to_geograph(stationlist.iloc[i].lat)
        elif station_ref == "geographic":
            latitude = stationlist.iloc[i].lat
        longitude = stationlist.iloc[i].lon
        station_id = network + '.' + station + '..' + channel
        print(station_id)
        if use_topo:
            elevation = stationlist.iloc[i].elev_m / 1000.
        else:
            elevation = 0.
        if use_buried:
            depth = stationlist.iloc[i].sensor_depth_m / 1000.
            depth = depth - elevation
        else:
            if not use_topo:
                depth = 0.
            else:
                depth = -elevation

        if source_type == "Gauss":
            if solver == "axisem3d":
                stftype = "gauss"
            elif solver == "specfem3d":
                stftype = 0
        elif source_type == "Ricker":
            if solver == "axisem3d":
                stftype = "ricker"
            elif solver == "specfem3d":
                stftype = 1

        for component in components:
            if solver == "axisem3d":
                eventfid = open(os.path.join(outdir,
                                             'CMTSOLUTION' + '_' + station +
                                             '_' + component), 'w')
            elif solver == "specfem3d":
                eventfid = open(os.path.join(outdir,
                                         'FORCESOLUTION' + '_' + station +
                                         '_' + component), 'w')
            eventfid.write('FORCE 001 \n')
            eventfid.write('time shift:    0.0000   \n')
            eventfid.write('half duration:    %s   \n' % str(hdur))
            eventfid.write('latitude:    %s   \n' % str(latitude))
            eventfid.write('longitude:    %s   \n' % str(longitude))
            eventfid.write('depth:    %s   \n' % str(depth))

            if component == 'Z':
                f = [0.0, 0.0, -1.0]
            elif component == 'E':
                f = [1.0, 0.0, 0.0]
            elif component == 'N':
                f = [0.0, 1.0, 0.0]

            if solver == "axisem3d":
                eventfid.write('source time function:  %s \n' % stftype)
                f = [fc * factor_force for fc in f]
                eventfid.write('Fp:   %g   \n' % f[0])
                eventfid.write('Ft:   %g   \n' % f[1])
                eventfid.write('Fr:   %g   \n' % f[2])

            elif solver == "specfem3d":
                eventfid.write('source time function:  %g \n' % stftype)
                eventfid.write('factor force source:       %g \n' % factor_force)
                eventfid.write('component dir vect source E:   %g   \n' % f[0])
                eventfid.write('component dir vect source N:   %g   \n' % f[1])
                eventfid.write('component dir vect source Z_UP:   %g   \n' % f[2])


if __name__ == '__main__':
    os.system("mkdir -p " + outdir)
    stations_to_forcesolutions(stationlist, hdur_pointsource, outdir, solver,
                               use_buried=use_buried, use_topo=use_topo,
                               factor_force=factor_force)
