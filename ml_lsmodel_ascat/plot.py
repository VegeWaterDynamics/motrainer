import numpy as np
import matplotlib.pyplot as plt
import cartopy
import shapely.geometry as sgeom
from copy import copy


def plot_gsdata(data,
                nrowcol,
                outpath,
                outformat='jpeg',
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
    subplotid = 0
    features = data.columns[2:]
    for ax in axes.flat:
        ax.set_extent(basemap_extent)
        ax.axis('equal')
        ax.add_feature(coast, lw=0.8, alpha=0.5)
        ax.add_feature(ocean, alpha=0.4)

        sc = ax.scatter(data['lon'].values,
                        data['lat'].values,
                        marker='o',
                        c=data[features[subplotid]],
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
        if subplotid % nrowcol[1] == 0:
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
            if subplotid % nrowcol[1] == nrowcol[1] - 1:
                axpos = ax.get_position()
                cbar_ax = fig.add_axes(
                    [axpos.x1, axpos.y0, 0.01, axpos.height])  # l, b, w, h
                cbar = fig.colorbar(sc, cax=cbar_ax)
                cbar.ax.tick_params(labelsize=fontsize / 2)

        # Title
        title_text = '(' + chr(97 + subplotid) + ') '  # Alphabet label
        if isinstance(titles, list):
            title_text = title_text + titles[subplotid]
        ax.set_title(label=title_text, fontsize=fontsize, loc='center')

        # Row label
        if rowlabels is not None:
            if cbar_mode == 'plot' or cbar_mode == 'row':
                if subplotid % nrowcol[1] == nrowcol[1] - 1:
                    cbar.set_label(rowlabels[int(subplotid / nrowcol[1])],
                                   fontsize=fontsize)
            else:
                if subplotid % nrowcol[1] == 0:
                    ax.set_ylabel(rowlabels[int(subplotid / nrowcol[1])],
                                  fontsize=fontsize)

        subplotid += 1

    # Colorbar for the entire figure
    if cbar_mode == 'fig':
        cax = fig.add_axes([0.35, 0.06, 0.3, 0.01])
        cax.tick_params(labelsize=fontsize / 2)
        plt.colorbar(sc, cax=cax, orientation='horizontal')
        if cbar_label is not None:
            cax.set_xlabel(cbar_label, fontdict={'size': fontsize})

    plt.savefig(outpath, bbox_inches='tight', format=outformat)


def plot_tsdata(data,
                nrowcol,
                outpath,
                outformat='jpeg',
                figsize=None,
                linecolor=None,
                titles=None,
                rowlabels=None,
                kw_padding=None,
                fontsize=8):
    """plot time-series data

    :param data: list of DataFrames. Each DataFrames in the list will
        be plotted into a subplot. The row index of DataFrame should be
        DatetimeIndex. The columns will be ploted
    :type data: list
    :param nrowcol: number of (row, cloumn) of the subplots.
    :type nrowcol: tuple
    :param outpath: output path
    :type outpath: str
    :param outformat: outformat, defaults to 'jpeg'
    :type outformat: str, optional
    :param figsize: figure size in inches, defaults to None
    :type figsize: tuple or list, optional
    :param linecolor: line color per column in data, defaults to None
    :type linecolor: dict, optional
    :param titles: Titles of each plot, defaults to None
    :type titles: list, optional
    :param rowlabels: row labels, defaults to None
    :type rowlabels: list, optional
    :param kw_padding: Keyword arguement for pyplot.tight_layout().
        Adjust subplot padding.
        E.g. {'pad': 5, 'h_pad': 5, 'w_pad': 3}. defaults to None.
    :type kw_padding: dict, optional
    :param fontsize: fontsize of tick, title and labels,
        defaults to 8.
    :type fontsize: int, optional
    """

    fig, axes = plt.subplots(nrows=nrowcol[0], ncols=nrowcol[1])

    if kw_padding is not None:
        plt.tight_layout(**kw_padding)

    if figsize is not None:
        fig.set_size_inches(figsize[0], figsize[1])

    subplotid = 0
    linelist = dict()
    for ax in axes.flat:
        # Plot
        for col in data[subplotid].columns:
            if (linecolor is not None) and (col in linecolor.keys()):
                lc = linecolor[col]
            else:
                lc = ['b', 'g', 'r', 'c', 'm', 'y',
                      'k'][data[subplotid].columns.get_loc(col)]
            line = ax.plot(data[subplotid][col], color=lc)
            linelist[col] = line[0]
            ax.tick_params(axis='both', labelsize=fontsize / 2)

        # Title
        title_text = '(' + chr(97 + subplotid) + ') '  # Alphabet label
        if isinstance(titles, list):
            title_text = title_text + titles[subplotid]
        ax.set_title(label=title_text, fontsize=fontsize, loc='center')

        # Row label
        if rowlabels is not None:
            if subplotid % nrowcol[1] == 0:
                ax.set_ylabel(rowlabels[int(subplotid / nrowcol[1])],
                              fontsize=fontsize)

        subplotid += 1

    # Legend
    fig.legend(handles=linelist.values(),
               labels=linelist.keys(),
               fontsize=fontsize,
               ncol=min([5, len(linelist)]),
               loc="lower center")
    fig.subplots_adjust(left=0.07,
                        right=0.93,
                        wspace=0.15,
                        hspace=0.35,
                        bottom=0.15)
    plt.savefig(outpath, bbox_inches='tight', format=outformat)


def _lambert_xticks(ax, ticks, tick_location, tickfontsize):
    """Draw ticks on the bottom x-axis of a Lambert Conformal projection."""
    xticks, xticklabels = _lambert_ticks(ax, ticks, tick_location)
    ax.xaxis.tick_bottom()
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [ax.xaxis.get_major_formatter()(xtick) for xtick in xticklabels])
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(tickfontsize)


def _lambert_yticks(ax, ticks, tick_location, tickfontsize):
    """Draw ricks on the left y-axis of a Lamber Conformal projection."""
    yticks, yticklabels = _lambert_ticks(ax, ticks, tick_location)
    ax.yaxis.tick_left()
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        [ax.yaxis.get_major_formatter()(ytick) for ytick in yticklabels])
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(tickfontsize)


def _lambert_ticks(ax, ticks, tick_location):
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
        if tick_location == 'left':
            xy = np.vstack((np.linspace(extent[0], extent[1],
                                        n_steps), np.zeros(n_steps) + t)).T
        elif tick_location == 'bottom':
            xy = np.vstack((np.zeros(n_steps) + t,
                            np.linspace(extent[2], extent[3], n_steps))).T
        proj_xyz = ax.projection.transform_points(cartopy.crs.Geodetic(),
                                                  xy[:, 0], xy[:, 1])
        xyt = proj_xyz[..., :2]
        ls = sgeom.LineString(xyt.tolist())
        locs = axis.intersection(ls)
        if not locs:
            tick = [None]
        else:
            if tick_location == 'left':
                tick = locs.xy[1]
            elif tick_location == 'bottom':
                tick = locs.xy[0]
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
