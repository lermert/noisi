# plotting on the map
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np


def nice_map(ax, lat_min, lat_max, lon_min, lon_max,
             proj=ccrs.PlateCarree,
             latitude_labels_west=True,
             latitude_labels_east=False,
             longitude_labels_south=True,
             longitude_labels_north=False,
             axislabelpad=-0.15):

    if proj not in [ccrs.Orthographic]:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj())

    try:
        gl = ax.gridlines(draw_labels=True)
        if not latitude_labels_east:
            gl.ylabels_right = False
        if not longitude_labels_north:
            gl.xlabels_top = False
        if not latitude_labels_west:
            gl.ylabels_left = False
        if not longitude_labels_south:
            gl.xlabels_bottom = False

        if latitude_labels_west:  # -0.15 or so for square plots
            ax.text(axislabelpad, 0.5, 'Latitude (°)',
                    va='bottom', ha='center',
                    rotation='vertical', rotation_mode='anchor',
                    transform=ax.transAxes, fontsize='large')
        if longitude_labels_south:
            ax.text(0.5, axislabelpad, 'Longitude (°)',
                    va='bottom', ha='center',
                    rotation='horizontal', rotation_mode='anchor',
                    transform=ax.transAxes, fontsize='large')
    except TypeError:
        pass


def plot_grid(map_x, map_y, map_z, stations=[], locations=[], v=None,
              globe=False,
              outfile=None, title=None, shade='flat', cmap=plt.cm.viridis,
              sequential=False, v_min=None, normalize=False,
              coastres='110m', proj=ccrs.PlateCarree, quant_unit='PSD',
              lat_0=None, lon_0=None, lon_min=None, lon_max=None,
              lat_min=None, lat_max=None, resol=1, alpha=1.0, size=None,
              axes=None, colorbar=True, stationnames=[], axislabelpad=-0.15,
              plot_coastlines=True, stationcolor='r', stasizefactor=0.3,
              colorbar_ticks=None):

    if lat_0 is None:
        lat_0 = 0.5 * (map_y.max() - map_y.min())
    if lon_0 is None:
        lon_0 = 0.5 * (map_x.max() - map_x.min())

    if lon_min is None:
        lon_min = np.min(map_x)
    if lon_max is None:
        lon_max = np.max(map_x)
    if lat_max is None:
        lat_max = np.max(map_y)
    if lat_min is None:
        lat_min = np.min(map_y)

    if resol != 1:
        map_x = map_x[::resol]
        map_y = map_y[::resol]
        map_z = map_z[::resol]

    if axes is None:
        fig = plt.figure(figsize=(5.4, 4.6))
        ax = fig.add_subplot(1, 1, 1, projection=proj())
        nice_map(ax, lat_min, lat_max, lon_min, lon_max, proj=proj,
                 axislabelpad=axislabelpad)
    else:
        ax = axes
    if title is not None:
        plt.title(title)

    if normalize:
        map_z /= np.abs(map_z).max()
        quant_unit = 'Normalized (-)'
    if v is None:
        v = np.max(map_z)
    if v_min is None:
        if sequential:
            v_min = 0.
        else:
            v_min = -v

    if size is None:
        size = 100. * abs(map_x[1] - map_x[0])
    scplt = ax.scatter(map_x, map_y, c=map_z,
                       alpha=alpha, marker='.',
                       cmap=cmap, s=size, vmin=v_min, vmax=v,
                       transform=ccrs.PlateCarree())
    if axes is None and colorbar:
        cbar = plt.colorbar(scplt)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label(quant_unit, rotation=270)
        if colorbar_ticks is not None:
            cbar.set_ticks(colorbar_ticks[0])
            cbar.set_ticklabels(colorbar_ticks[1])

    if plot_coastlines:
        ax.coastlines(resolution=coastres, linewidth=1.)

    # draw station locations
    if len(stations) == len(stationnames):
        for i, stan in enumerate(stationnames):
            sta = stations[i]
            ax.text(sta[1] + 0.01 * size, sta[0] + 0.01, stan,
                    bbox=dict(facecolor='0.7', alpha=0.5))
    for sta in stations:
        ax.plot(sta[1], sta[0], '^', color=stationcolor,
                markersize=stasizefactor * size,
                zorder=4)
    for loc in locations:
        ax.plot(loc[1], loc[0], 'x', color='lightgreen', markersize=0.5 * size)
    if axes is None:
        if outfile is None:
            plt.show()
        else:
            plt.savefig(outfile, dpi=300.)
            plt.close()
    else:
        return(scplt)


def plot_sourcegrid(gridpoints, coastres='110m', size=None,
                    proj=ccrs.PlateCarree, globe=False, outfile=None,
                    **kwargs):

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(1, 1, 1, projection=proj())
    plt.title('Source grid', fontsize='x-large')

    lat_min = gridpoints[1].min()
    lat_max = gridpoints[1].max()
    lon_min = gridpoints[0].min()
    lon_max = gridpoints[0].max()
    nice_map(ax, lat_min, lat_max, lon_min, lon_max, proj=proj)

    if size is None:
        size = 100. * abs(gridpoints[0, 1] - gridpoints[0, 0])
    ax.scatter(gridpoints[0], gridpoints[1], s=size, marker='.',
               transform=ccrs.PlateCarree(), **kwargs)
    ax.coastlines(resolution=coastres)

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, dpi=300.)
        plt.close()


def plot_window(correlation, window, measurement):

    maxlag = correlation.stats.npts * correlation.stats.delta
    lag = np.linspace(-maxlag, maxlag, correlation.stats.npts)

    plt.plot(lag, correlation.data / np.max(np.abs(correlation.data)))
    plt.plot(lag, window / np.max(np.abs(window)), '--')
    plt.title(correlation.id)

    plt.text(0, -0.75, 'Measurement value: %s' % str(round(measurement, 3)))
    plt.xlabel('Correlation Lag in seconds.')
    plt.ylabel('Normalized correlation and window.')

    plt.show()
