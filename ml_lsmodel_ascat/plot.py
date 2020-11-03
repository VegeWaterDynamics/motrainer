import numpy as np
import matplotlib.pyplot as plt
import cartopy
import shapely.geometry as sgeom
from copy import copy


def plot_gsdata(data,
                nrowcol,
                titles=None,
                rowlabels=None,
                kw_padding=None,
                buffer_percentage=0.05,
                basemap_scale='50m',
                basemap_extent='auto',
                marksize=5,
                fontsize=8,
                colormap='YlGnBu',
                cbar_mode='plot',
                cbar_label=None):
    """Plot geo-spratial data on a basmap

    :param data: Input data for plotting. First two columns
        should be ['lat', 'lon']. Each other column will be
        plotted orderly as a subplot.
    :type data: pandas.DataFrame
    :param nrowcol: number of (row, cloumn) of the subplots.
    :type nrowcol: tuple
    :param titles: Titles of each plot, defaults to None
    :type titles: list, optional
    :param rowlabels: row labels, defaults to None
    :type rowlabels: list, optional
    :param kw_padding: Keyword arguement for pyplot.tight_layout().
        Adjust subplot padding.
        E.g. {'pad': 5, 'h_pad': 5, 'w_pad': 3}. defaults to None.
    :type kw_padding: dict, optional
    :param buffer_percentage: The fraction of the extra basemap
        extent to the data extent. when >0, the basemap will be
        larger than the data coverage., defaults to 0.05
    :type buffer_percentage: float, optional
    :param basemap_scale: Scale of the basemap elements,
        defaults to '50m'
    :type basemap_scale: str, optional
    :param basemap_extent: Mannual basemap extent setting,
        defaults to 'auto', which means the extent will be
        determined based on the data coverage, and the `buffer_percentage`
    :type basemap_extent: list, optional
    :param marksize: scatter plot maker size, defaults to 5
    :type marksize: int, optional
    :param fontsize: fontsize of tick, title and labels,
        defaults to 8. For colorbar the tick labe size will be halved.
    :type fontsize: int, optional
    :param colormap: Mannual color scale setting, defaults to 'YlGnBu'
    :type colormap: str, optional
    :param cbar_mode: Modes of colorbar. Choose from 'plot'
        (colorbar per plot), 'row' (colorbar per row) or
        'fig' (one colorbar for the entire figure), defaults to 'plot'
    :type cbar_mode: str, optional
    :param cbar_label: Labels for colorvar. Only active when cbar_mode
        is 'fig'. defaults to None
    :type cbar_label: str, optional
    """

    if basemap_extent == 'auto':
        buffer_lon = (max(data['lon'].values) -
                      min(data['lon'].values)) * buffer_percentage
        buffer_lat = (max(data['lat'].values) -
                      min(data['lat'].values)) * buffer_percentage
        basemap_extent = [
            min(data['lon'].values) - buffer_lon,
            max(data['lon'].values) + buffer_lon,
            min(data['lat'].values) - buffer_lat,
            max(data['lat'].values) + buffer_lat,
        ]

    data_crs = cartopy.crs.PlateCarree()
    plot_crs = cartopy.crs.Orthographic(
        central_longitude=(basemap_extent[3] - basemap_extent[2]) / 2.,
        central_latitude=(basemap_extent[1] - basemap_extent[0]) / 2.,
        globe=None)

    coast = cartopy.feature.NaturalEarthFeature(category='physical',
                                                scale=basemap_scale,
                                                name='coastline',
                                                facecolor='none',
                                                edgecolor='k')
    ocean = cartopy.feature.NaturalEarthFeature(category='physical',
                                                scale=basemap_scale,
                                                name='ocean',
                                                facecolor='#DDDDDD')

    fig, axes = plt.subplots(nrows=nrowcol[0],
                             ncols=nrowcol[1],
                             subplot_kw={'projection': plot_crs})
    if kw_padding is not None:
        plt.tight_layout(**kw_padding)
    feature_id = 0
    features = data.columns[2:]
    for ax in axes.flat:
        ax.set_extent(basemap_extent)
        ax.axis('equal')
        ax.add_feature(coast, lw=0.8, alpha=0.5)
        ax.add_feature(ocean, alpha=0.4)

        sc = ax.scatter(data['lon'].values,
                        data['lat'].values,
                        marker='o',
                        c=data[features[feature_id]],
                        transform=data_crs,
                        cmap=colormap,
                        s=5)

        # Ticks and gridlines
        xticks = np.arange(
            np.floor(basemap_extent[0]), np.ceil(basemap_extent[1]),
            np.floor(basemap_extent[1] - basemap_extent[0]) / 4.).tolist()
        yticks = np.arange(
            np.floor(basemap_extent[2]), np.ceil(basemap_extent[3]),
            np.floor(basemap_extent[3] - basemap_extent[2]) / 4.).tolist()
        ax.gridlines(xlocs=xticks, ylocs=yticks)
        ax.xaxis.set_major_formatter(cartopy.mpl.gridliner.LONGITUDE_FORMATTER)
        ax.yaxis.set_major_formatter(cartopy.mpl.gridliner.LATITUDE_FORMATTER)
        if feature_id % nrowcol[1] == 0:
            _lambert_yticks(ax, yticks, 'left', fontsize)
        _lambert_xticks(ax, xticks, 'bottom', fontsize)

        # Colorbar
        if cbar_mode == 'plot':
            axpos = ax.get_position()
            cbar_ax = fig.add_axes([axpos.x1, axpos.y0, 0.0075,
                                    axpos.height])  # l, b, w, h
            cbar = fig.colorbar(sc, cax=cbar_ax)
            cbar.ax.tick_params(labelsize=fontsize / 2)
        elif cbar_mode == 'row':
            if feature_id % nrowcol[1] == nrowcol[1] - 1:
                axpos = ax.get_position()
                cbar_ax = fig.add_axes(
                    [axpos.x1, axpos.y0, 0.01, axpos.height])  # l, b, w, h
                cbar = fig.colorbar(sc, cax=cbar_ax)
                cbar.ax.tick_params(labelsize=fontsize / 2)

        # Title
        title_text = '(' + chr(97 + feature_id) + ') '  # Alphabet label
        if isinstance(titles, list):
            title_text = title_text + titles[feature_id]
        ax.set_title(label=title_text, fontsize=fontsize, loc='center')

        # Row label
        if rowlabels is not None:
            if cbar_mode == 'plot' or cbar_mode == 'row':
                if feature_id % nrowcol[1] == nrowcol[1] - 1:
                    cbar.set_label(rowlabels[int(feature_id / nrowcol[1])],
                                   fontsize=fontsize)
            else:
                if feature_id % nrowcol[1] == 0:
                    ax.set_ylabel(rowlabels[int(feature_id / nrowcol[1])],
                                  fontsize=fontsize)

        feature_id += 1

    # Colorbar for the entire figure
    if cbar_mode == 'fig':
        cax = fig.add_axes([0.35, 0.06, 0.3, 0.01])
        cax.tick_params(labelsize=fontsize / 2)
        plt.colorbar(sc, cax=cax, orientation='horizontal')
        if cbar_label is not None:
            cax.set_xlabel(cbar_label, fontdict={'size': fontsize})

    plt.savefig('testfig.jpg', bbox_inches='tight', format='jpeg')


def _lambert_xticks(ax, ticks, tick_location, tickfontsize):
    """Draw ticks on the bottom x-axis of a Lambert Conformal projection."""
    te = lambda xy: xy[0]
    lc = lambda t, n, b: np.vstack(
        (np.zeros(n) + t, np.linspace(b[2], b[3], n))).T
    xticks, xticklabels = _lambert_ticks(ax, ticks, tick_location, lc, te)
    ax.xaxis.tick_bottom()
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [ax.xaxis.get_major_formatter()(xtick) for xtick in xticklabels])
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(tickfontsize)


def _lambert_yticks(ax, ticks, tick_location, tickfontsize):
    """Draw ricks on the left y-axis of a Lamber Conformal projection."""
    te = lambda xy: xy[1]
    lc = lambda t, n, b: np.vstack(
        (np.linspace(b[0], b[1], n), np.zeros(n) + t)).T
    yticks, yticklabels = _lambert_ticks(ax, ticks, tick_location, lc, te)
    ax.yaxis.tick_left()
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        [ax.yaxis.get_major_formatter()(ytick) for ytick in yticklabels])
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(tickfontsize)


def _lambert_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
    """Get the tick locations and labels
    for an axis of a Lambert Conformal projection."""
    xb = ax.get_xbound()
    yb = ax.get_ybound()
    outline_patch = sgeom.LineString([[xb[0], yb[0]], [xb[0], yb[1]],
                                      [xb[1], yb[1]], [xb[1], yb[0]]])
    axis = _find_side(outline_patch, tick_location)
    extent = ax.get_extent(cartopy.crs.PlateCarree())
    n_steps = 30
    _ticks = []
    for t in ticks:
        xy = line_constructor(t, n_steps, extent)
        proj_xyz = ax.projection.transform_points(cartopy.crs.Geodetic(),
                                                  xy[:, 0], xy[:, 1])
        xyt = proj_xyz[..., :2]
        ls = sgeom.LineString(xyt.tolist())
        locs = axis.intersection(ls)
        if not locs:
            tick = [None]
        else:
            tick = tick_extractor(locs.xy)
        _ticks.append(tick[0])
    # Remove ticks that aren't visible:
    ticklabels = copy(ticks)
    while True:
        try:
            index = _ticks.index(None)
        except ValueError:
            break
        _ticks.pop(index)
        ticklabels.pop(index)
    return _ticks, ticklabels


def _find_side(ls, side):
    """
    Given a shapely LineString which is assumed to be rectangular, return the
    line corresponding to a given side of the rectangle.
    """
    minx, miny, maxx, maxy = ls.bounds
    points = {
        'left': [(minx, miny), (minx, maxy)],
        'right': [(maxx, miny), (maxx, maxy)],
        'bottom': [(minx, miny), (maxx, miny)],
        'top': [(minx, maxy), (maxx, maxy)],
    }
    return sgeom.LineString(points[side])
