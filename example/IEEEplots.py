"""
Created on Fri Feb 21 16:04:26 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:03:35 2019

Inspired by:
https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b    
https://github.com/llSourcell/visualize_dataset_demo/blob/master/data_visualization.py

Paper:

http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?source=post_page

https://lvdmaaten.github.io/tsne/
---------------------------
@author: Xu Shan

plot for IEEE
"""
# %matplotlib qt5
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

import shapely.geometry as sgeom
from copy import copy

import matplotlib as mpl # THIS IS FOR REMOTE FIGURING
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']#指定默认字体
mpl.rcParams['text.usetex'] = False
#
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os, glob

from sklearn.neighbors.kde import KernelDensity
from sklearn.manifold import TSNE

from tensorflow.keras.models import load_model

from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import matthews_corrcoef
from math import sqrt  
import scipy as scipy    
from scipy import stats
from scipy.stats.stats import pearsonr, spearmanr

import seaborn as sns
import matplotlib.lines as mlines


import matplotlib.transforms as mtransforms

import seaborn as sns
from glob import glob
import iris
from shapely.geometry import Point


import os

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM

# Keras items
from keras import regularizers

from keras.optimizers import Adam, Nadam, Adagrad, sgd
from keras.activations import relu, elu, tanh

from keras.losses import mean_squared_error, mean_squared_logarithmic_error,mean_absolute_error

from keras import backend as K
from keras.models import load_model

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from keras import regularizers

from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations


from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import matthews_corrcoef
from math import sqrt  
import scipy as scipy    
from scipy import stats

import dill

large_fontsize = 20
small_fontsize = 15
smallest_fotsz = 10
# https://www.ieee.org/content/dam/ieee-org/ieee/web/org/pubs/format_def_tbl.pdf

import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates

import cartopy
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import rasterio
import geopandas as gpd
from rasterio.mask import mask


xmajorLocator   = MultipleLocator(2)
ymajorLocator   = MultipleLocator(2)
xminorLocator   = MultipleLocator(1)
yminorLocator   = MultipleLocator(1)

large_fontsize = 20
small_fontsize = 15
smallest_fotsz = 10
#%%
# https://www.ieee.org/content/dam/ieee-org/ieee/web/org/pubs/format_def_tbl.pdf

shp_dir = 'E:/Surfsara Output DNN/ref-nuts-2016-20m.shp'

#France_shp = gpd.read_file(shp_dir+"/fr_10km.shp")
nuts = gpd.read_file(shp_dir+'/NUTS_RG_20M_2016_4326_LEVL_0.shp/NUTS_RG_20M_2016_4326_LEVL_0.shp')
country = nuts[nuts['NUTS_ID'] == 'FR']
def geom_to_masked_cube(df, geometry, lats, lons,
                        mask_excludes=True):
    """
    Convert a shapefile geometry into a mask for a cube's data.
    
    Args:
    
    * cube:
        The cube to mask.
    * geometry:
        A geometry from a shapefile to define a mask.
    * x_coord: (str or coord)
        A reference to a coord describing the cube's x-axis.
    * y_coord: (str or coord)
        A reference to a coord describing the cube's y-axis.
    
    Kwargs:
    
    * mask_excludes: (bool, default False)
        If False, the mask will exclude the area of the geometry from the
        cube's data. If True, the mask will include *only* the area of the
        geometry in the cube's data.
    
    .. note::
        This function does *not* preserve lazy cube data.
    
    """
    # Get horizontal coords for masking purposes.

#    lats = stat.lat.values
#    lons = stat.lon.values
#    lon2d, lat2d = np.meshgrid(lons,lats)
#
#    # Reshape to 1D for easier iteration.
#    lon2 = lon2d.reshape(-1)
#    lat2 = lat2d.reshape(-1)
    
    mask_t = []
    # Iterate through all horizontal points in cube, and
    # check for containment within the specified geometry.
    for lat, lon in zip(lats, lons):
#        this_point = gpd.geoseries.Point(lon, lat)
        this_point = Point(lon, lat)
        res = geometry.contains(this_point)
        mask_t.append(res.values[0])
    
    mask_t = np.array(mask_t)#.reshape(lon2d.shape)
    if mask_excludes:
        # Invert the mask if we want to include the geometry's area.
        mask_t = ~mask_t
#    # Make sure the mask is the same shape as the cube.
#    dim_map = (cube.coord_dims(y_coord)[0],
#               cube.coord_dims(x_coord)[0])
#    cube_mask = iris.util.broadcast_to_shape(mask_t, cube.shape, dim_map)
    
    # Apply the mask to the cube's data.
    df_copy = df.copy()
    data = df_copy.values
    masked_data = np.ma.masked_array(data, mask_t)
    df_copy = masked_data
    return df_copy
#%%
def geom_to_masked_2D(df, geometry, lats, lons,
                        mask_excludes=True):
    """
    Convert a shapefile geometry into a mask for a 2D's data.
    
    Args:
    
    * cube:
        The cube to mask.
    * geometry:
        A geometry from a shapefile to define a mask.
    * x_coord: (str or coord)
        A reference to a coord describing the cube's x-axis.
    * y_coord: (str or coord)
        A reference to a coord describing the cube's y-axis.
    
    Kwargs:
    
    * mask_excludes: (bool, default False)
        If False, the mask will exclude the area of the geometry from the
        cube's data. If True, the mask will include *only* the area of the
        geometry in the cube's data.
    
    .. note::
        This function does *not* preserve lazy cube data.
    
    """
    # Get horizontal coords for masking purposes.

#    lats = stat.lat.values
#    lons = stat.lon.values
#    lon2d, lat2d = np.meshgrid(lons,lats)
#
#    # Reshape to 1D for easier iteration.
#    lon2 = lon2d.reshape(-1)
#    lat2 = lat2d.reshape(-1)
    
    mask_t = np.zeros([df.shape[0], df.shape[1]])
    # Iterate through all horizontal points in cube, and
    # check for containment within the specified geometry.
    for lat, lon in itertools.product(lats, lons):
#        this_point = gpd.geoseries.Point(lon, lat)
        this_point = Point(lon, lat)
        if mask_excludes:
            res = ~geometry.contains(this_point)
        else:
            res = geometry.contains(this_point)
        mask_t[np.where(lats == lat), np.where(lons == lon)] = res
        #mask_t.append(res.values[0])
    
    mask_t = np.array(mask_t)#.reshape(lon2d.shape)
#    if mask_excludes:
#        # Invert the mask if we want to include the geometry's area.
#        mask_t = ~mask_t
#    # Make sure the mask is the same shape as the cube.
#    dim_map = (cube.coord_dims(y_coord)[0],
#               cube.coord_dims(x_coord)[0])
#    cube_mask = iris.util.broadcast_to_shape(mask_t, cube.shape, dim_map)
    
    # Apply the mask to the cube's data.
    df_copy = df.copy()
    data = df_copy#.values
    masked_data = np.ma.masked_array(data, mask_t)
    df_copy = masked_data
    return df_copy, mask_t
#%%
def find_side(ls, side):
    
    """
    Given a shapely LineString which is assumed to be rectangular, return the
    line corresponding to a given side of the rectangle.
    
    """
    minx, miny, maxx, maxy = ls.bounds
    points = {'left': [(minx, miny), (minx, maxy)],
              'right': [(maxx, miny), (maxx, maxy)],
              'bottom': [(minx, miny), (maxx, miny)],
              'top': [(minx, maxy), (maxx, maxy)],}
    return sgeom.LineString(points[side])


def lambert_xticks(ax, ticks, tick_location):
    """Draw ticks on the bottom x-axis of a Lambert Conformal projection."""
    te = lambda xy: xy[0]
    lc = lambda t, n, b: np.vstack((np.zeros(n) + t, np.linspace(b[2], b[3], n))).T
    xticks, xticklabels = _lambert_ticks(ax, ticks, tick_location, lc, te)
    ax.xaxis.tick_bottom()
    ax.set_xticks(xticks)
    ax.set_xticklabels([ax.xaxis.get_major_formatter()(xtick) for xtick in xticklabels])
    for tick in ax.xaxis.get_major_ticks():  
        tick.label1.set_fontsize(smallest_fotsz)
    

def lambert_yticks(ax, ticks,tick_location):
    """Draw ricks on the left y-axis of a Lamber Conformal projection."""
    te = lambda xy: xy[1]
    lc = lambda t, n, b: np.vstack((np.linspace(b[0], b[1], n), np.zeros(n) + t)).T
    yticks, yticklabels = _lambert_ticks(ax, ticks, tick_location, lc, te)
    ax.yaxis.tick_left()
    ax.set_yticks(yticks)
    ax.set_yticklabels([ax.yaxis.get_major_formatter()(ytick) for ytick in yticklabels])
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(smallest_fotsz)

def _lambert_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
    """Get the tick locations and labels for an axis of a Lambert Conformal projection."""
    outline_patch = sgeom.LineString(ax.outline_patch.get_path().vertices.tolist())
    axis = find_side(outline_patch, tick_location)
    n_steps = 30
    extent = ax.get_extent(ccrs.PlateCarree())
    _ticks = []
    for t in ticks:
        xy = line_constructor(t, n_steps, extent)
        proj_xyz = ax.projection.transform_points(ccrs.Geodetic(), xy[:, 0], xy[:, 1])
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
#%%
def keras_dnn(learning_rate, num_dense_layers,num_input_nodes,
              num_dense_nodes, activation):
    model = Sequential()
#start the model making process and create our first layer
    
    model.add(Dense(num_input_nodes, 
                input_shape=(4,), 
                activation=activation
                ))
    #Notice that there the input_shape is important!!!!!!
    #create a loop making a new dense layer for the amount passed to this model.
    #naming the layers helps avoid tensorflow error deep in the stack trace.
    #no pooling/conv because not a figure processing!
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i+1)
        model.add(Dense(num_dense_nodes,
                        activation=activation,
                        name=name
                        ))
        #add our classification layer.
    model.add(Dense(units = 3))
        
    #setup our optimizer and compile
    #adam = Adam(lr=learning_rate)
    #model.compile(optimizer=adam, loss= mean_squared_error,
    #              metrics=['mae', 'acc'])
    return model
    
def get_data_labels(data):
    data_norm = pd.DataFrame()
    data_norm['LAI'] = data['LAI'].values
    data_norm['WG2'] = data['WG2'].values
    data_norm['GPP'] = data['GPP'].values
    data_norm['RE'] = data['RE'].values
        
        #data_norm['slope']= data['slope'].values
#        data_norm['curv']= data['curv'].values
        
    label_norm = pd.DataFrame()
    label_norm['sig']= data['sig'].values
    #for a model which can produce 3 paras simutaneously
    label_norm['slope']= data['slope'].values
    label_norm['curv']= data['curv'].values
        
    lables = label_norm.values
    input_set = data_norm.values
        
    return lables, input_set
def pre_normalization(data, pre_Method):
    output = []
    #min_max = MinMaxScaler()
    #pre_Method = StandardScaler()
    x = pre_Method.fit_transform((data.values).reshape(-1,1))
    #x = stand.fit_transform((data.values).reshape(-1,1))
    for i in range(len(x)):
        y = x[i]
        output.append(y)

    return output
def class_select(df,lonmax, lonmin, latmax, latmin):
    df = df[cluster_stat['lon']<=lonmax]
    df = df[cluster_stat['lon']>=lonmin]
    df = df[cluster_stat['lat']<=latmax]
    df = df[cluster_stat['lat']>=latmin]
    return df
#%% change the ROI

#%%
color1 = 'darkgreen'
color2 = 'darkgreen'
color3 = 'midnightblue'
color4 = 'forestgreen'
color5 = 'darkgrey'
color6 = 'salmon'
color7 ='peru'

colors = []
for i in range(7):
    colors.append(locals()['color'+str(i+1)])
#%%
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
# Plotting function
df = open('E:/Surfsara Output DNN/DNN Run Output Pickle/France/df_stat_xs', 'rb')

clusters = pickle.load(df)
df.close()
stat  = clusters[0]
del clusters
#%%
#%%
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
# subplot of RMSE and Pearson Corr coef of sigma, backscatter, curvature
# notice that the fontsize of label should be modified!
# Saving figurelatmin = min(lat)
# Fig 3
def subPlotMap(Z, lon, lat,figname,save_at,var_stat, sub_title, cb_tick): 
    latmax = 52
    lonmin =-5
    lonmax = 10
    latmin = 41#min(lat)
    

    
    img_extent = [lonmin, lonmax, latmin, latmax]
    
    # Projection system
    data_crs=cartopy.crs.PlateCarree()
    proj=cartopy.crs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=proj))
    
    # Colorbar limits
    #cbmax = 6
    #cbmin = 1
    
    # Data and axes
    #x, y = np.meshgrid(lon,lat)
    #fig = plt.figure() #1, (4., 4.)
    #ax_pcs = fig.add_subplot(2,3,i+1)
    #fig.set_size_inches(7.25, 6)
#    ax = AxesGrid(fig, 111, axes_class=axes_class,
#                  axes_pad=0.3,  # pad between axes in inch.
#              nrows_ncols=(2, 3), label_mode='')
    coast = cartopy.feature.NaturalEarthFeature(
            category='physical', scale='50m', 
            name='coastline', facecolor='none', edgecolor='k')
    #    land = cartopy.feature.NaturalEarthFeature(
    #            category='physical', scale='10m', name='coastline', facecolor='none')
    ocean = cartopy.feature.NaturalEarthFeature(
            category='physical', scale='50m', 
            name='ocean', facecolor='#DDDDDD')
    #    borders = cartopy.feature.NaturalEarthFeature(
    #            category='cultural', scale='10m',
    #            name='admin_0_boundary_lines_land', facecolor='none', edgecolor='black')
    #projection = ccrs.epsg(32636)
    fig, axes = plt.subplots(nrows=2,ncols=3,
                             subplot_kw={'projection': cartopy.crs.Orthographic(central_longitude=0,
                                        central_latitude=45, globe=None)} 
                             #subplot_kw={'projection': ccrs.PlateCarree()} 
                             # https://stackoverflow.com/questions/51621362/subplot-attributeerror-axessubplot-object-has-no-attribute-get-extent
                             )
    fig.set_size_inches(7, 6)
    plt.tight_layout(pad = 5.28, h_pad=3.08, w_pad=2)
    i = 0
    for ax in axes.flat:
        #ax = fig.add_subplot(2,3,i+1)
        #        ax = AxesGrid(fig, 231+i, axes_class=axes_class,
#                  axes_pad=0.5,  # pad between axes in inch.
#              nrows_ncols=(1, 1), label_mode='')
        # if this method is used, do not add the crrs in plt.subplots!
#
        ax.set_extent(img_extent)
        #ax[0].add_feature(land, lw=0.4, alpha=0.7, zorder=1)
        ax.add_feature(coast, lw=0.8, alpha=0.5, zorder=9)
        ax.add_feature(ocean, alpha=0.4,zorder=8)
        #ax[0].add_feature(borders, lw=0.4, zorder=2)  
        cmap = mpl.colors.ListedColormap(["grey","darkgreen","lightgreen", "darkred", "darkblue", "lime", "goldenrod", "olive",'darksalmon', 'yellowgreen', 'skyblue', 'khaki','dimgray'])
        
        #temp = pd.DataFrame(Z[var_stat[i]])
        temp = geom_to_masked_cube(Z[var_stat[i]], country, lat, lon, mask_excludes=True)
        if i<=2:
            sc = ax.scatter(lon, lat, marker='o', c=temp, transform=data_crs, 
                            cmap='YlGnBu',
                               zorder=7,
                               #vmin=0.75,vmax=0.9, 
                               s=5) #zorder decides layer order (1 is bottom)
            ax.set_title(label = '('+chr(97+i)+') '+sub_title[i], fontsize = smallest_fotsz, loc = 'center')
        if i>=3:
            sc = ax.scatter(lon, lat, marker='o', c=temp, transform=data_crs, 
                            cmap='YlGnBu',
                               zorder=7,
                               vmin=0.65,vmax=0.9, 
                               s=5) #zorder decides layer order (1 is bottom)
            ax.set_title(label = '('+chr(97+i)+') '+sub_title[i-3], fontsize = smallest_fotsz, loc = 'center')
        
        fig.canvas.draw()
        yticks = [latmin-2+i*3 for i in range(latmax-latmin)]#[-110, -50, -40, -30, -20, -11, 0, 10, 20, 30, 40, 50]
        xticks = [lonmin-2+i*3 for i in range(lonmax-lonmin)]#[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        ax.gridlines(xlocs=xticks, ylocs=yticks)
        ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
        ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
        if i == 0 or i == 3:
            lambert_yticks(ax, yticks, 'left')
        lambert_xticks(ax, xticks, 'bottom')
        # Gridlines
#        gl = ax.gridlines(crs=proj, draw_labels=True,
#                      linewidth=1.5, color='black', alpha=0.65, zorder=10,linestyle='--')
#                    #alpha sets opacity
#        gl.xlabels_top = False #no xlabels on top (longitudes)
#        gl.ylabels_right = False #no ylabels on right side (latitudes)
#        if i != 0 and i != 3:
#            gl.ylabels_left = False
        if i<= 2:
            axpos=ax.get_position()
            cbar_ax=fig.add_axes([axpos.x1, axpos.y0,0.0075, axpos.height]) #l, b, w, h
#            divider = make_axes_locatable(ax[0])
#            cax = divider.append_axes("right", size="5%", pad=0.05)
#
#            cbar=fig.colorbar(sc, cax=cax)
            cbar=fig.colorbar(sc,cax=cbar_ax)
            cbar.ax.tick_params(labelsize=smallest_fotsz/2)
            if i == 2:
                cbar.set_label(cb_tick[0], fontsize = smallest_fotsz)#'Pearson correlation coefficient')
        elif i == 5:
            axpos=ax.get_position()
            cbar_ax=fig.add_axes([axpos.x1, axpos.y0,0.0075, axpos.height])
            cbar=fig.colorbar(sc,cax=cbar_ax)
            cbar.ax.tick_params(labelsize=smallest_fotsz/2)
            cbar.set_label(cb_tick[1], fontsize = smallest_fotsz)#'Pearson correlation coefficient')
        #gl.xlines = False
        #gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
#        gl.xlocator = xmajorLocator
#        gl.ylocator = ymajorLocator
#        ax.xaxis.set_minor_locator(xminorLocator)  
#        ax.yaxis.set_minor_locator(yminorLocator)  
#        
#        ax.xaxis.grid(True, which='major', linestyle='--') #x - major  
#        ax.yaxis.grid(True, which='minor', linestyle='--') #y - major
#        
#        gl.xformatter = LONGITUDE_FORMATTER
#        gl.yformatter = LATITUDE_FORMATTER
#        gl.xlabel_style = {'size': smallest_fotsz, 'color': 'black',
##                           'weight': 'bold',
#                           'rotation': 45} #formatting of gridline labels
#        gl.ylabel_style = {'size': smallest_fotsz, 
##                           'weight': 'bold',
#                           'color': 'black'}
        
        make_axes_locatable(ax) #divider = 
        i = i+1
    fig.show()
#    plt.savefig('{}{}'.format(save_at,figname),bbox_inches = 'tight',format='jpeg')
    plt.savefig('{}{}'.format(save_at,figname+'.eps'),
                bbox_inches = 'tight',
                format='eps')
    plt.savefig('{}{}'.format(save_at,figname+'.jpeg'),
                bbox_inches = 'tight',
                format='jpeg')
    
    # Colorbar

#    cbar.set_label('RMSE'+figname)
var_stat = ['RMSE_sig', 'RMSE_slope', 'RMSE_curv', 'r_sig', 'r_slope', 'r_curv']
#sub_title= ['sigma','slope','curvature']
sub_title= [r'$\sigma_{40}^o$ [dB]',r"$\sigma^{\prime}$ [dB/deg]",r'$\sigma^{\prime\prime}{\rm[dB/deg^2]}$']

cb_tick = ['RMSE', 'Pearson correlation \n coefficient']
save_at = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/'
subPlotMap(stat, stat['lon'], stat['lat'],"Fig3_performance_ortho",save_at+'IEEEplots/',
           var_stat, sub_title, cb_tick)
    
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
# subplot of RMSE and Pearson Corr coef of sigma, backscatter, curvature
# notice that the fontsize of label should be modified!
# Saving figurelatmin = min(lat)
# end of Fig 3


#%%
def subPlotMap_general(Z, lon, lat,figname,save_at,var_stat, sub_title, cb_tick): 
    latmax = 52
    lonmin =-5
    lonmax = 10
    latmin = 41#min(lat)
    
    img_extent = [lonmin, lonmax, latmin, latmax]
    
    # Projection system
    data_crs=cartopy.crs.PlateCarree()
    proj=cartopy.crs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=proj))
    
    data_crs_1 = cartopy.crs.Orthographic(central_longitude=0,
                                        central_latitude=45, globe=None)
#    proj = cartopy.crs.Orthographic(central_longitude=0,
#                                        central_latitude=45, globe=None)
#    axes_class = (GeoAxes, dict(map_projection=proj))
    
    coast = cartopy.feature.NaturalEarthFeature(
            category='physical', scale='50m', 
            name='coastline', facecolor='none', edgecolor='k')
    ocean = cartopy.feature.NaturalEarthFeature(
            category='physical', scale='50m', 
            name='ocean', facecolor='#DDDDDD')
    fig, axes = plt.subplots(nrows=2,ncols=3, #ccrs.PlateCarree()
                             subplot_kw={'projection': cartopy.crs.Orthographic(central_longitude=0,
                                        central_latitude=45, globe=None)} 
                             # https://stackoverflow.com/questions/51621362/subplot-attributeerror-axessubplot-object-has-no-attribute-get-extent
                             )
    fig.set_size_inches(7, 6)
    plt.tight_layout(pad = 5.28, h_pad=3.08, w_pad=2) #pad = 5.28, h_pad=3.08, w_pad=2 # 7, 0.5, 3
    i = 0
    for ax in axes.flat:
        ax.set_extent(img_extent, crs=ccrs.PlateCarree())
        #ax[0].add_feature(land, lw=0.4, alpha=0.7, zorder=1)
        ax.add_feature(coast, lw=0.8, alpha=0.5, zorder=9)
        ax.add_feature(ocean, alpha=0.4,zorder=8)
        #ax[0].add_feature(borders, lw=0.4, zorder=2)  
        cmap = mpl.colors.ListedColormap(["grey","darkgreen","lightgreen", "darkred", "darkblue", "lime", "goldenrod", "olive",'darksalmon', 'yellowgreen', 'skyblue', 'khaki','dimgray'])
        
        temp = geom_to_masked_cube(Z[var_stat[i]], country, lat, lon, mask_excludes=True)
        
        sc = ax.scatter(lon, lat, marker='o', c=temp, transform=data_crs, 
                        cmap='YlGnBu',
                        #zorder=7,
                       #vmin=0.75,vmax=0.9, 
                        s=5) #zorder decides layer order (1 is bottom)
        ax.set_title(label = '('+chr(97+i)+') '+sub_title[i%3], fontsize = smallest_fotsz, loc = 'center')
        
        fig.canvas.draw()
        yticks = [latmin-2+i*3 for i in range(latmax-latmin)]#[-110, -50, -40, -30, -20, -11, 0, 10, 20, 30, 40, 50]
        xticks = [lonmin-2+i*3 for i in range(lonmax-lonmin)]#[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        ax.gridlines(xlocs=xticks, ylocs=yticks)
        ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
        ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
        if i == 0 or i == 3:
            lambert_yticks(ax, yticks, 'left')
        lambert_xticks(ax, xticks, 'bottom')
#        gl = ax.gridlines(crs=data_crs, draw_labels=True,
#                          zorder=10,
#                          linestyle='--', linewidth=.5, color = 'black')
        #ax.gridlines(crs=data_crs_1, which='major', axis = 'y', linestyle='--', linewidth=.5, alpha=.3)
        
        
#        gl = ax.gridlines(crs=proj, draw_labels=True,
#                      linewidth=0.5, color='black', alpha=0.65, zorder=10,linestyle='--')
#                    #alpha sets opacity
#        gl.xlabels_top = False #no xlabels on top (longitudes)
#        gl.ylabels_right = False #no ylabels on right side (latitudes)
#        if i != 0 and i != 3:
#            gl.ylabels_left = False
#        gl.xlocator = xmajorLocator
#        gl.ylocator = ymajorLocator
#        
#        gl.xformatter = LONGITUDE_FORMATTER
#        gl.yformatter = LATITUDE_FORMATTER
#        gl.xlabel_style = {'size': smallest_fotsz, 'color': 'black',
##                           'weight': 'bold',
#                           'rotation': 45} #formatting of gridline labels
#        gl.ylabel_style = {'size': smallest_fotsz, 
##                           'weight': 'bold',
#                           'color': 'black'}
#        
        axpos=ax.get_position()
        cbar_ax=fig.add_axes([axpos.x1, axpos.y0,0.0075, axpos.height])#/(23/12)]) #l, b, w, h

        cbar=fig.colorbar(sc,cax=cbar_ax)
        cbar.ax.tick_params(labelsize=smallest_fotsz/2)
        if i == 2 or i == 5:
            cbar.set_label(cb_tick[int(i/3)], fontsize = smallest_fotsz)#'Pearson correlation coefficient')
        
        make_axes_locatable(ax) #divider = 
        i = i+1
    fig.show()
#    plt.savefig('{}{}'.format(save_at,figname),bbox_inches = 'tight',format='jpeg')
    plt.savefig('{}{}'.format(save_at,figname+'.eps'),bbox_inches = 'tight',format='eps')
    plt.savefig('{}{}'.format(save_at,figname+'.jpg'),bbox_inches = 'tight',format='jpeg')
    # Colorbar
xmajorLocator   = MultipleLocator(2)
ymajorLocator   = MultipleLocator(2)
xminorLocator   = MultipleLocator(1)
yminorLocator   = MultipleLocator(1)

var_stat = ['mean_sig', 'mean_slope', 'mean_curv', 'std_sig', 'std_slope', 'std_curv']
cb_tick = ['mean', 'standard deviation']
sub_title= [r'$\sigma_{40}^o$ [dB]',r"$\sigma^{\prime}$ [dB/deg]",r'$\sigma^{\prime\prime}{\rm[dB/deg^2]}$']
save_at = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/'

subPlotMap_general(stat, stat['lon'], stat['lat'],"Fig1_SpatialPattern_ortho",save_at+'IEEEplots/',
           var_stat, sub_title, cb_tick)
#%%
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
# Plotting function
df = open('E:/Surfsara Output DNN/DNN Run Output Pickle/France/df_long_tsne_shapley', 'rb')

clusters = pickle.load(df)
df.close()
df_long  = clusters [0] # with shapley value!
df_long.columns
del clusters
#%%
max_types_series = []
precent_veg = []
veg_types = [4,7,10,15]
for i in range(len(veg_types)):
    france_index = geom_to_masked_cube(df_long['Vegtype{}'.format(veg_types[i])], country, df_long.lat, df_long.lon, mask_excludes=True)


#    stored = df_long[df_long['Vegtype{}'.format(veg_types[i])]==np.max(df_long['Vegtype{}'.format(veg_types[i])])]
    stored = df_long[df_long['Vegtype{}'.format(veg_types[i])]==np.max(france_index)]
    
    
    #stored = df_long[df_long['Vegtype{}'.format(veg_types[i])]==np.max(df_long['Vegtype{}'.format(veg_types[i])])]
    max_types_series.append(stored)
    print(stored.index[0])
#%%
var_stat = ['RMSE_sig', 'RMSE_slope', 'RMSE_curv', 'r_sig', 'r_slope', 'r_curv']
var_stat1 = ['mean_sig', 'mean_slope', 'mean_curv', 'std_sig', 'std_slope',
            'std_curv']
ref_name = ['lon','lat','cluster','value','RMSE_sig', 'RMSE_slope', 'RMSE_curv', 'r_sig', 'r_slope', 'r_curv',
            'mean_sig', 'mean_slope', 'mean_curv', 'std_sig', 'std_slope',
            'std_curv', 'range_sig', 'range_slope', 'range_curv']
veg_name = ['Vegtype4', 'Vegtype7', 'Vegtype10', 'Vegtype15']
#veg_class = ['broadleaf', 'agriculture', 'grass', 'needleleaf']
veg_class = ["M.C. Broadleaf", "N. Agric.", "M.C. Grasslands", "Les Landes"]
cluster_stat = pd.DataFrame()
for i in range(len(ref_name)):
    if i<=3:
        hehe = df_long
    elif i>3:
        hehe = stat
    cluster_stat[ref_name[i]] = geom_to_masked_cube(hehe[ref_name[i]], country, stat['lat'], stat['lon'], mask_excludes=True)
for column in veg_name:
    cluster_stat[column] = geom_to_masked_cube(df_long[column], country, stat['lat'], stat['lon'], mask_excludes=True)

criteria = [0.35, 0.50, 0.50, 0.40]
listCluster = cluster_stat['cluster'].unique()
listCluster.sort()
listVeg = []
#cluster_stat['cluster'].unique()
for i, veg in enumerate(veg_name):
    listVeg.append(cluster_stat[cluster_stat[veg] >= criteria[i]])
#listVeg.sort()

stat_veg = pd.DataFrame()
var_stat_total=  var_stat+var_stat1
for j in range(len(var_stat_total)):
    data_temp = []
    for i in range(len(veg_name)):
        temp = cluster_stat[cluster_stat[veg_name[i]] >= criteria[i]][var_stat_total[j]]
        if i == 0:
            temp = class_select(temp, 3, 0, 46.5, 44)
        if i == 1:
            temp = class_select(temp, 5, 0.2, 51, 47.5)
        if i == 2:
            temp = class_select(temp, 5.4, 0, 48, 43)
#        data_temp.append(cluster_stat[cluster_stat[veg_name[i]] >= criteria[i]][var_stat_total[j]])
        data_temp.append(temp)
    
    stat_veg[var_stat_total[j]] = data_temp        
#%%
colors = ['brown','salmon','darkblue','green']
def plotting_time_types(max_types_series,
                        pick_year,
                        input_name, var_name, fig_name, 
                        unit, GPI, save_at):
    # using all of the variables to plot a 3*4 subuplot
    # var_name (3) x cluster_name (4)
    
    fig, ax = plt.subplots(nrows=3,ncols=4)
    fig.set_size_inches(28, 12) # 30, 10 # 7，6
    #plt.tight_layout(pad = 1.08, h_pad=1.08, w_pad=None)
    i = 0
    for axes in ax.flat:
        cluster_ind = i%4
        var_ind = int(i/4)
        lon = max_types_series[cluster_ind]['data'].iloc[0]['lon'][0]
        lat = max_types_series[cluster_ind]['data'].iloc[0]['lat'][0]
        line_list = []
#        im = ax_pcs.imshow(temp.resample('M').mean().dropna(axis=0,how='any').transpose().values,
#                           vmin=0, vmax=1)
        for input_ind in range(len(input_name)):
            if pick_year == 0:
                temp_year = max_types_series[cluster_ind]['shap_{}'.format(var_name[var_ind])].iloc[0][input_name[input_ind]]
                temp_year.index = pd.to_datetime(temp_year.index)
                line = axes.plot(temp_year.groupby(temp_year.index.dayofyear).mean(), 
                      #label =input_name[input_ind]+unit[var_ind],
                          color=colors[len(colors)-input_ind-1], linewidth=1)
                
                axes.set_xlabel('DOY', fontsize=small_fontsize+2)#smallest_fotsz)
                axes.xaxis.set_major_locator(MultipleLocator(30*2)) 
                # MultipleLocator(30)
            else:
                temp_year = max_types_series[cluster_ind]['shap_{}'.format(var_name[var_ind])].iloc[0][input_name[input_ind]]
                temp_year.index = pd.to_datetime(temp_year.index)
                temp_year = temp_year[temp_year.index.year == pick_year].rolling(window = 5).mean().dropna().resample('5D').first()
                line = axes.plot(temp_year, 
                      #label =input_name[input_ind]+unit[var_ind],
                          color=colors[len(colors)-input_ind-1], linewidth=1)
                
                axes.set_xlabel('year of '+str(pick_year), fontsize=small_fontsize+2)#smallest_fotsz)
                
                monthFmt = mdates.DateFormatter('%b')
                axes.xaxis.set_major_locator(mdates.MonthLocator())
                axes.xaxis.set_major_formatter(monthFmt)
                
#                axes.set_xticks(temp_year[temp_year.index.year == pick_year].index)
#                axes.set_xticklabels(temp_year.groupby(temp_year.index.dayofyear).mean().index, rotation=45)
            line_list.append(line)
        for tick in axes.xaxis.get_major_ticks():  
            tick.label1.set_fontsize(small_fontsize)
        for tick in axes.yaxis.get_major_ticks():  
            tick.label1.set_fontsize(small_fontsize)
        if cluster_ind == 0:
            axes.set_ylabel('Feature Importance \n Fraction [-] ('+sub_title[var_ind]+')',
                            fontsize=small_fontsize+2)#smallest_fotsz)
        axes.set_ylim([0, 1])
        
        temp = max_types_series[cluster_ind]['shap_{}'.format(var_name[var_ind])].iloc[0]
        temp.index = pd.to_datetime(temp.index)
        #axes.xaxis.set_major_locator(xmajorLocator) 
        
        if var_ind == 0:
            axes.set_title('('+chr(97+i)+') {} - {}°N {}°E'.format(GPI[cluster_ind],lat, lon),
                           fontsize = small_fontsize+2)#smallest_fotsz)
        else:
            axes.set_title('('+chr(97+i)+')',
                           fontsize = small_fontsize+2)#smallest_fotsz)
        # add legend outside the subplots
        #lg_ax = fig.add_axes([0.15, 0.15, 0.7, 0.05])
        
        i = i+1
    #h,l = axes.get_legend_handles_labels()
    lgd = fig.legend(line_list[0:4],     # The line objects [line1, line2, line3, line4]
               labels=input_name,   # The labels for each line
               fontsize = small_fontsize+2,
               borderaxespad=0.1,    # Small spacing around legend box
               ncol=4,
               borderpad=1.5, labelspacing=1.5,
               loc="upper center",
               bbox_to_anchor=(0.5-0.07/2,0.1-0.01))
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.subplots_adjust(left=0.07, right=0.93, wspace=0.15, hspace=0.35,
                        bottom = 0.15)
    plt.savefig('{}{}'.format(save_at,fig_name+'.eps'),bbox_extra_artists=(lgd,), bbox_inches='tight',format='eps')
    plt.savefig('{}{}'.format(save_at,fig_name+'.jpeg'),bbox_extra_artists=(lgd,), bbox_inches='tight',format='jpeg')
    
    plt.show()
    #plt.close()


input_name = ['LAI','WG2','GPP','RE']
var_name = ['sig', 'slope', 'curv']
sub_title= [r'$\sigma_{40}^o$ [dB]',r"$\sigma^{\prime}$ [dB/deg]",r'$\sigma^{\prime\prime}{\rm[dB/deg^2]}$']

GPI = ['Broadleaf', 'Agriculture','Grassland', 'Needleleaf']
veg_class = ["M.C. Broadleaf", "N. Agric.", "M.C. Grasslands", "Les Landes"]

unit = ['[dB]', '[dB/deg]', '[dB/deg]'] # double check the unit for curv!
save_at = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/IEEEplots/'

xmajorLocator   = MultipleLocator(30)#MultipleLocator(2*365)
ymajorLocator   = MultipleLocator(2)
xminorLocator   = MultipleLocator(1)
yminorLocator   = MultipleLocator(1)

plotting_time_types(max_types_series, 0, input_name, var_name, 
                    "Fig5_feature_importance_climatology", 
                        unit, GPI, save_at)
#pick_year = 2015
#plotting_time_types(max_types_series, pick_year, input_name, var_name, 
#                    "Fig5_feature_importance_pentad_"+str(pick_year), 
#                        unit, GPI, save_at)
#pick_year = 2017
#plotting_time_types(max_types_series, pick_year, input_name, var_name, 
#                    "Fig5_feature_importance_pentad_"+str(pick_year), 
#                        unit, GPI, save_at)
#%% Fig 6
# map the feature importance every month across the domain
# can use Manuel's codes!

#%% 
path = "E:/Surfsara Output DNN/DNN Run Output Pickle/"

df = open(path+'dataframe_per_GPI_including_sig_slope_curv_LAI_WG2_plus_VEGETATIONTYPE_RE_GPP_NEW', 'rb')
df_all_gpi = pickle.load(df)
df.close()
df_all_gpi = df_all_gpi[0]
#%% Fig 2
# t-sne results <-> high resolution data
# need further work!!!!!
# processing the high resolution data of ESA!!!!
import xarray as xr
from netCDF4 import Dataset
import h5netcdf.legacyapi as netCDF4

ESA_NC = Dataset("E:/Surfsara Output DNN/ESA.nc")
#ESA_NC = Dataset("E:/Surfsara Output DNN/C3S-LC-L4-LCCS-Map-300m-P1Y-2017-v2.1.1.nc")
#ESA_NC = netCDF4.Dataset("E:/Surfsara Output DNN/C3S-LC-L4-LCCS-Map-300m-P1Y-2017-v2.1.1.nc")
print(ESA_NC.variables.keys())
XX = ESA_NC.variables['lon'][:] #129600
XY = ESA_NC.variables['lat'][:] #64800 # lccs_class

XX = XX[:].reshape((1,XX[:].size))
XY = XY[:].reshape((1,XY[:].size))

XX = XX[0,:]
XY = XY[0,:]

lon = XX.data
lat = XY.data

latmax = 52
lonmin =-5
lonmax = 10
latmin = 41#min(lat)

lonmin_ind, lonmax_ind = np.where(lon>= lonmin)[0][0], np.where(lon<= lonmax)[0][-1]
latmin_ind, latmax_ind = np.where(lat>= latmin)[0][-1], np.where(lat<= latmax)[0][0]

ESA_class = ESA_NC.variables['lccs_class'][:,latmax_ind:latmin_ind,lonmin_ind:lonmax_ind] # 1 64800 129600
lon_class = lon[lonmin_ind:lonmax_ind]
lat_class = lat[latmax_ind:latmin_ind]

ESA_class = np.array(ESA_class.reshape([lat_class.size, lon_class.size]), 'float64')
ESA_NC.close()
#%%
path = "E:/Surfsara Output DNN/DNN Run Output Pickle/France/"

df = open(path+'Dominant_Vegtype_and_Fraction_per_GPI', 'rb')
df_veg_gpi = pickle.load(df)
df.close()
df_veg_gpi = df_veg_gpi[0]
#%%
def coaser_resolution_2D(pop_density, coarseness):
    # Suppose the size of pop_density was 198x147 instead of 200x150.
    # Start by finding the next highest multiple of 5x5
    shape = np.array(pop_density.shape, dtype=float)
    new_shape = coarseness * np.ceil(shape / coarseness).astype(int)
    # new_shape is now (200, 150)
    
    # Create the zero-padded array and assign it with the old density
    zp_pop_density = np.zeros(new_shape)
    zp_pop_density[:int(shape[0]), :int(shape[1])] = pop_density
    
    # Now use the same method as before
    temp = zp_pop_density.reshape((new_shape[0] // coarseness, coarseness,
                                   new_shape[1] // coarseness, coarseness))
    coarse_pop_density = np.mean(temp, axis=(1,3)) # or np.sum
    return coarse_pop_density
def coaser_resolution_1D(pop_density, coarseness):
    # Suppose the size of pop_density was 198x147 instead of 200x150.
    # Start by finding the next highest multiple of 5x5
    shape = np.array(pop_density.shape, dtype=float)
    new_shape = coarseness * np.ceil(shape / coarseness).astype(int)
    # new_shape is now (200, 150)
    
    # Create the zero-padded array and assign it with the old density
    zp_pop_density = np.zeros(new_shape)
    zp_pop_density[:int(shape[0])] = pop_density
    
    # Now use the same method as before
    temp = zp_pop_density.reshape((new_shape[0] // coarseness, coarseness))
    coarse_pop_density = np.mean(temp, axis=(1)) # or np.sum
    return coarse_pop_density
ESA_class_40 = coaser_resolution_2D(ESA_class, coarseness = 40)
lat_class_40 = coaser_resolution_1D(lat_class, coarseness = 40)
lon_class_40 = coaser_resolution_1D(lon_class, coarseness = 40)
#%% FIG 2

def landCoverClass(Z, lon, lat,figname,save_at,var_stat, sub_title, cb_tick): 
    latmax = 52
    lonmin =-5
    lonmax = 10
    latmin = 41#min(lat)
    
    img_extent = [lonmin, lonmax, latmin, latmax]
    
    # Projection system
    data_crs=cartopy.crs.PlateCarree()
    proj=cartopy.crs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=proj))
    coast = cartopy.feature.NaturalEarthFeature(
            category='physical', scale='50m', 
            name='coastline', facecolor='none', edgecolor='k')
    
    ocean = cartopy.feature.NaturalEarthFeature(
            category='physical', scale='50m', 
            name='ocean', facecolor='#DDDDDD')
    
    fig, axes = plt.subplots(nrows=2,ncols=2,
                             subplot_kw={'projection': cartopy.crs.Orthographic(central_longitude=0,
                                        central_latitude=45, globe=None)} 
                             #subplot_kw={'projection': ccrs.PlateCarree()} 
                             # https://stackoverflow.com/questions/51621362/subplot-attributeerror-axessubplot-object-has-no-attribute-get-extent
                             )
    fig.set_size_inches(7.25, 7.25)
    plt.tight_layout(pad = 3.28, h_pad=4, w_pad=None)
    i = 0
    for ax in axes.flat:
        if i in [0,1,2]:
            ax.set_extent(img_extent)
        #ax[0].add_feature(land, lw=0.4, alpha=0.7, zorder=1)
            ax.add_feature(coast, lw=0.8, alpha=0.5, zorder=9)
            ax.add_feature(ocean, alpha=0.4,zorder=8)
        #ax[0].add_feature(borders, lw=0.4, zorder=2)  
        
        if i == 0:
            temp = geom_to_masked_cube(Z[var_stat[i]], country, lat, lon, mask_excludes=True)
            if i == 0:
                cmap = plt.get_cmap('YlGn', np.max(temp)-np.min(temp)+1)
            else:
                cmap = 'YlGn' ####need to change!!!!!!!!!!!!!!!!!!!!!!!
            sc = ax.scatter(lon, lat, marker='o', c=temp, transform=data_crs, cmap=cmap,#'RdYlGn',
                        zorder=7,
                        s=15) #zorder decides layer order (1 is bottom)
        elif i == 1:
            temp = geom_to_masked_cube(Z[var_stat[1]], country, lat, lon, mask_excludes=True)
            if i == 0:
                cmap = plt.get_cmap('YlGn', np.max(temp)-np.min(temp)+1)
            else:
                cmap = 'YlGn' ####need to change!!!!!!!!!!!!!!!!!!!!!!!
            sc = ax.scatter(lon, lat, marker='o', c=temp, transform=data_crs, cmap=cmap,#'RdYlGn',
                        zorder=7,
                        s=15) #zorder decides layer order (1 is bottom)
        elif i == 2: #df_veg_gpi
            my_veg_types = {'Vegtype1': "no vegetation (smooth)",
                            'Vegtype2': " no vegetation (rocks)",
                            'Vegtype3': "permanent snow\nand ice",
                            'Vegtype4': "temperate broadleaf\ncold-deciduous\nsummergreen",
                            'Vegtype5': "boreal needleleaf\nevergreen",
                            'Vegtype6': "tropical broadleaf\nevergreen",
                            'Vegtype7': "C3 cultures types",
                            'Vegtype8': "C4 cultures types",
                            'Vegtype9': "irrigated crops", 
                            'Vegtype10': "grassland (C3)",
                            'Vegtype11': "tropical grassland\n(C4)",
                            'Vegtype12': "irrigated grass",
                            'Vegtype13': "tropical broadleaf\ndeciduous",
                            'Vegtype14': "temperate broadleaf\nevergreen",
                            'Vegtype15': "temperate needleleaf\nevergreen",
                            'Vegtype16': "boreal broadleaf\ncold-deciduous\nsummergreen",
                            'Vegtype17': " boreal needleleaf\ncold-deciduous\nsummergreen",
                            'Vegtype18': "boreal grass",
                            'Vegtype19': "shrub - SHRB"}

#            cmap = mpl.colors.ListedColormap(["grey","grey","grey",'darkgreen',"lightgreen", 
#                                              'darkgreen',"darkred","red", "blue", "lime", 
#                                              "lime", "lime",'darkgreen', 'olive', 'yellowgreen',
#                                              'springgreen', "lightgreen",'khaki','dimgray'])
            cmap_list = ["#ffecdd","#a7a7a7","#ffffff",'#005500',"#347003", 
                         '#005500',"darkred","red", "#aaffff", "#ffaa00", 
                         "#ffaa00", "#ffaa00",'#005500', 'olive', 'yellowgreen',
                         '#55aa00', "#347003",'#03c79d','#825700']
            cmap = mpl.colors.ListedColormap(cmap_list)
            temp = geom_to_masked_cube(df_veg_gpi['Number_Name_Type'], country, lat, lon, mask_excludes=True)
            
            sc = ax.scatter(lon, lat, marker='o', c=temp, transform=data_crs, cmap=cmap,#'RdYlGn',
                        zorder=7,
                       #vmin=0.75,vmax=0.9, 
                        s=15) #zorder decides layer order (1 is bottom)
            
#            cmap = customColourMap#'YlGn'####need to change!!!!!!!!!!!!!!!!!!!!!!!
#            temp, mask_temp = geom_to_masked_2D(ESA_class_40, country, lat_class_40, lon_class_40, mask_excludes=True)
#            x, y = np.meshgrid(lon_class_40,lat_class_40)
#            temp[temp.mask] = np.nan
#            print('==============mask completed=================')
#            sc = ax.scatter(x, y, marker='.',c=temp.data, transform=data_crs, cmap=cmap)#,#'RdYlGn',
                        #zorder=7,
                       #vmin=0.75,vmax=0.9, 
                        #s=15) 
        blue_line = []
        if i == 3:
            for l in range(len(cmap_list)):
                blue_line.append(mlines.Line2D([], [], color=cmap_list[l],
                          markersize=5, label=my_veg_types['Vegtype'+str(l+1)], linewidth=4))
            axpos=ax.get_position()
            cbar_ax=fig.add_axes([axpos.x0, axpos.y0,axpos.x1-axpos.x0, axpos.height]) #l, b, w, h
            cbar_ax.axis('off')
#            cbar_ax.tick_params(axis='both',          # changes apply to the x-axis
#                                which='both',      # both major and minor ticks are affected
#                                bottom=False,      # ticks along the bottom edge are off
#                                left=False,      # ticks along the bottom edge are off
#                                top=False,         # ticks along the top edge are off
#                                labelbottom=False) # labels along the bottom edge are off)
            cbar_ax.legend(handles=blue_line, prop={'size': 7}, 
                           loc='lower left',
                           bbox_to_anchor=(-0.05,-0.075),
                           borderpad=1, labelspacing=1.6,
                           title="colorbar of (c)",
                           #bbox_to_anchor=(1., 1), 
                           shadow=True, ncol=2)
            
        if i in [0,1]:
            axpos=ax.get_position()
            cbar_ax=fig.add_axes([axpos.x1, axpos.y0,0.0075, axpos.height]) #l, b, w, h
            cbar=fig.colorbar(sc,cax=cbar_ax)
            cbar.ax.tick_params(labelsize=smallest_fotsz)
        if i in [0,1,2]:
            ax.set_title(label = '('+chr(97+i)+') '+sub_title[i], fontsize = smallest_fotsz, loc = 'center')
        
            fig.canvas.draw()
            yticks = [latmin-2+i*3 for i in range(latmax-latmin)]#[-110, -50, -40, -30, -20, -11, 0, 10, 20, 30, 40, 50]
            xticks = [lonmin-2+i*3 for i in range(lonmax-lonmin)]#[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
            ax.gridlines(xlocs=xticks, ylocs=yticks)
            ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
            ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
            if i == 0 or i == 2:
                lambert_yticks(ax, yticks, 'left')
            if i in [1,2]:
                lambert_xticks(ax, xticks, 'bottom')
        
        make_axes_locatable(ax) #divider = 
        i = i+1
        print('=============fig '+str(i)+'===============')
    fig.show()
    plt.savefig('{}{}'.format(save_at,figname+'.eps'),bbox_inches = 'tight',format='eps')
    plt.savefig('{}{}'.format(save_at,figname+'.jpeg'),bbox_inches = 'tight',format='jpeg')
    # Colorbar
xmajorLocator   = MultipleLocator(2)
ymajorLocator   = MultipleLocator(2)
xminorLocator   = MultipleLocator(1)
yminorLocator   = MultipleLocator(1)
var_stat = ['cluster', 'Max_Value']#['value', 'Max_Value', 'cluster']
cb_tick = ['t_SNE values', 'max value', 'cluster']#['t_SNE values', 'max value', 'cluster']
sub_title= ['t_SNE value cluster\n(Perplexity 70) ',
            'fraction of dominat\ncover type',
            'dominant vegetation\ncover type',]
veg_type = ['Vegtype'+str(i+1) for i in range(19)]
df_long_temp = df_long[veg_type]
max_type_list = []
for i in range(len(df_long)):
    df_long_temp = df_long[veg_type]
    max_type = max(df_long_temp.iloc[i])
    max_type_list.append(max_type)
    #if len(df_all_gpi[i].index) > 1:
        #max_type_list.append(df_all_gpi[i]['Max_Value'][df_all_gpi[i].index[0]])
df_long['Max_Value'] = max_type_list

cluster_vegtype = pd.DataFrame()
save_at_fig2 = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/IEEEplots/'

ESA_colorbar_path = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/IEEEplots/cci_lc_legend.txt'
ESA_colorbar = []
with open(ESA_colorbar_path, 'r') as file:
    lines = file.readlines()
    #data = file.read().replace('\n', '').split(';')
for line in lines:
    ESA_colorbar.append(line.strip().replace('\n', '').split(';'))
#
#data_text = np.loadtxt(ESA_colorbar_path, dtype={'name': ('id', 'name_1', 'color'),
#          'formats': (np.float, '<S100', '|S7')}, delimiter=';', skiprows=1)

from matplotlib.colors import LinearSegmentedColormap
cMap = []
#for value, colour in zip([0,1,5,10,25,50,100,500,1000],["White", "DarkGreen", "LightGreen", "Yellow", "Brown", "Orange", "IndianRed", "DarkRed", "Purple"]):
#    cMap.append((value/1000.0, colour))
for id_i, name, colour in ESA_colorbar:
    if id_i != 'id':
        cMap.append((int(id_i)/220.0, colour))

customColourMap = LinearSegmentedColormap.from_list("custom", cMap)

landCoverClass(df_long, df_long['lon'], df_long['lat'],"Fig2_LandCover_3subplots_ortho_40",save_at_fig2,
           var_stat, sub_title, cb_tick)

#%%
#import the new main clusters
path = "E:/Surfsara Output DNN/DNN Run Output Pickle/France/"

df = open(path+'max_classes_broadleaf_agriculture_grassland_needleleaf', 'rb')
df_frac = pickle.load(df)
df.close()
df_frac = df_frac[0]
#%% Fig 4
#
# boxplot of the performance with regard to the clusters
# group subboxplots? or 2*3 subplots...
def subboxplot(Z, figname,save_at,var_stat, sub_title, cb_tick): 
    
    fig, axes = plt.subplots(nrows=2,ncols=3
                             #subplot_kw={'projection': ccrs.PlateCarree()} 
                             # https://stackoverflow.com/questions/51621362/subplot-attributeerror-axessubplot-object-has-no-attribute-get-extent
                             )
    fig.set_size_inches(7, 6) #10, 6
    #fig.subplots_adjust(left=0.2)
    plt.tight_layout(
            pad = 2, 
            h_pad=3, w_pad=1.75)
    i = 0
    for ax in axes.flat:
        
        temp = Z[var_stat[i]]
        boxprops = dict(linestyle='-', 
                        linewidth=1,
                        #widths = .25,
                        alpha=.5,
                        color='k')
        whiskerprops = dict(linewidth = 1,
                            color = 'k',
                            alpha = .5)
        flierprops = dict(marker='o', markerfacecolor='grey',
                          alpha=.2,
                          #widths = .25,
                          #linewidth = 1,
                          markersize=1, linestyle='none',
                          fillstyle='full'
                          #markeredecolor='grey',
                          #markeredgewidth=0.0
                          )
        ax.boxplot(temp, 
                   widths = .4,
                   boxprops = boxprops,
                   whiskerprops = whiskerprops,
                   flierprops = flierprops)
        if i<=2:
            ax.set_title(label = '('+chr(97+i)+') '+sub_title[i], fontsize = smallest_fotsz, loc = 'center')
        if i>=3:
            ax.set_title(label = '('+chr(97+i)+') '+sub_title[i-3], fontsize = smallest_fotsz, loc = 'center')
        
        ax.xaxis.set_minor_locator(xminorLocator)  
        ax.yaxis.set_minor_locator(yminorLocator)
        ax.xaxis.set_ticklabels(veg_class)
        ax.tick_params(axis='x', rotation = 25)
#        if i > 2:
#            ax.set_xlabel('vegetation class', fontsize=smallest_fotsz)
        if i == 0 or i == 3:
            ax.set_ylabel(cb_tick[int(i/3)], fontsize=smallest_fotsz)
        for tick in ax.xaxis.get_major_ticks():  
            tick.label1.set_fontsize(smallest_fotsz-3)
        for tick in ax.yaxis.get_major_ticks():  
            tick.label1.set_fontsize(smallest_fotsz)

        make_axes_locatable(ax) #divider = 
        i = i+1
    fig.show()
#    plt.savefig('{}{}'.format(save_at,figname+'.eps'),bbox_inches = 'tight',format='eps')
#    plt.savefig('{}{}'.format(save_at,figname+'.jpeg'),bbox_inches = 'tight',format='jpeg')#, dpi = 3000)
    
    # Colorbar

#    cbar.set_label('RMSE'+figname)
var_stat = ['RMSE_sig', 'RMSE_slope', 'RMSE_curv', 'r_sig', 'r_slope', 'r_curv']
var_stat1 = ['mean_sig', 'mean_slope', 'mean_curv', 'std_sig', 'std_slope',
            'std_curv']
var_stat2 = ['lat', 'lon']
#sub_title= ['sigma','slope','curvature']
sub_title= [r'$\sigma_{40}^o$ [dB]',r"$\sigma^{\prime}$ [dB/deg]",r'$\sigma^{\prime\prime}{\rm[dB/deg^2]}$']
cb_tick = ['RMSE', 'Pearson correlation \n coefficient']
cb_tick1 = ['mean', 'standard deviation']
save_at = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/'
ref_name = ['lon','lat','cluster','value','RMSE_sig', 'RMSE_slope', 'RMSE_curv', 'r_sig', 'r_slope', 'r_curv',
            'mean_sig', 'mean_slope', 'mean_curv', 'std_sig', 'std_slope',
            'std_curv', 'range_sig', 'range_slope', 'range_curv']
veg_name = ['Vegtype4', 'Vegtype7', 'Vegtype10', 'Vegtype15']
#veg_class = ['broadleaf', 'agriculture', 'grass', 'needleleaf']
veg_class = ["M.C. Broadleaf", "N. Agric.", "M.C. Grasslands", "Les Landes"]
cluster_stat = pd.DataFrame()
for i in range(len(ref_name)):
    if i<=3:
        hehe = df_long
    elif i>3:
        hehe = stat
    cluster_stat[ref_name[i]] = geom_to_masked_cube(hehe[ref_name[i]], country, stat['lat'], stat['lon'], mask_excludes=True)
for column in veg_name:
    cluster_stat[column] = geom_to_masked_cube(df_long[column], country, stat['lat'], stat['lon'], mask_excludes=True)

criteria = [0.35, 0.50, 0.50, 0.40]
listCluster = cluster_stat['cluster'].unique()
listCluster.sort()
listVeg = []
#cluster_stat['cluster'].unique()
for i, veg in enumerate(veg_name):
    listVeg.append(cluster_stat[cluster_stat[veg] >= criteria[i]])
#listVeg.sort()


#stat_cluster = pd.DataFrame()
#
#for j in range(len(var_stat)):
#    data_temp = []
#    for i in range(len(listCluster)):
#        data_temp.append(cluster_stat[cluster_stat['cluster'].isin([listCluster[i]])][var_stat[j]])
#    stat_cluster[var_stat[j]] = data_temp

stat_veg = pd.DataFrame()
var_stat_total=  var_stat+var_stat1+var_stat2
for j in range(len(var_stat_total)):
    data_temp = []
    for i in range(len(veg_name)):
        temp = cluster_stat[cluster_stat[veg_name[i]] >= criteria[i]][var_stat_total[j]]
        if i == 0:
            temp = class_select(temp, 3, 0, 46.5, 44)
        if i == 1:
            temp = class_select(temp, 5, 0.2, 51, 47.5)
        if i == 2:
            temp = class_select(temp, 5.4, 0, 48, 43)
        if i == 3:
            temp = class_select(temp, 1, -2, 48, 43.5)
#        data_temp.append(cluster_stat[cluster_stat[veg_name[i]] >= criteria[i]][var_stat_total[j]])
        data_temp.append(temp)
    
    stat_veg[var_stat_total[j]] = data_temp
        
#subboxplot(stat_cluster,"Fig4_boxplot",save_at+'IEEEplots/',
#           var_stat, sub_title, cb_tick)
subboxplot(stat_veg,"Fig4_boxplot_vegclass",save_at+'IEEEplots/',
           var_stat, sub_title, cb_tick)
subboxplot(stat_veg,"Fig4_boxplot_meanStd",save_at+'IEEEplots/',
           var_stat1, sub_title, cb_tick1)
#%%
with open('E:/Surfsara Output DNN/DNN Run Output Pickle/France/df_stat_veg_manuel', 'wb') as f:
    pickle.dump([stat_veg], f)
#%%
import glob
GoodIndex_vegClass = [[1036, 1036, 1036], [1828, 1828, 1922], [788, 1236, 1383], [737, 289, 17]]
BadIndex_vegClass = [[1679, 270, 1893], [961, 680, 761], [1394, 1887, 1394], [168, 168, 168]]

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
def best_todate(nums):
    nums_list = list(nums)
    bt_nums = []
    for i in range(len(nums_list)):
        bt_nums.append(min(nums_list[0:i+1]))
    return np.array(bt_nums)

dimensions = ["learning rate",
              "num dense layers", #
              "num input nodes", #
              "num dense nodes", #
              "activation method",
              "batch size"]
nums_iter = [i for i in range(15)]


def convergence_plot(Z, var_name, cluster_name, unit, fig_name, save_at):
    plt.figure()
    fig, ax = plt.subplots(nrows=3,ncols=2)
    fig.set_size_inches(7.25, 10) #10, 6
    #fig.subplots_adjust(right=0.6)
    plt.tight_layout(
            pad = 4, 
            h_pad=5, w_pad=10)
    i = 0
    for axes in ax.flat:
        var_ind = int(i/2)
        goodbad_ind = int(i%2)
        goodbad = ['Good', 'bad']
        
        index = Z[goodbad_ind][var_ind]
        new_folder = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/GoodBad/GPI_'+str(index)+'_'+goodbad[goodbad_ind]+'/'
        GPI = 'GPI_'+str(index)+'_'+goodbad[goodbad_ind]
        filename = glob.glob(new_folder+'Best_gp_result_*')[0]
        print(filename)
        with open(filename, 'rb') as in_strm:
            gp_loaded = dill.load(in_strm)
        #gp_loaded = dill.load('{}gp_result_{}_{}'.format(new_folder,rmse_min_year,GPI))
        best_gp = gp_loaded
        
        
        par1 = axes.twinx() # num dense layers
        par2 = axes.twinx() # num input nodes
        par3 = axes.twinx() # num dense nodes
        # Offset the right spine of par2.  The ticks and label have already been
        # placed on the right by twinx above.
        par2.spines["right"].set_position(("axes", 1.2))
        par3.spines["right"].set_position(("axes", 1.4))
        # Having been created by twinx, par2 has its frame off, so the line of its
        # detached spine is invisible.  First, activate the frame but make the patch
        # and spines invisible.
        make_patch_spines_invisible(par2)
        make_patch_spines_invisible(par3)
        # Second, show the right spine.
        par2.spines["right"].set_visible(True)
        par3.spines["right"].set_visible(True)
        
        p1, = axes.plot(nums_iter, best_todate(best_gp.func_vals), "b-", label="RMSE")
        p2, = par1.plot(nums_iter, np.array(list(np.int_(np.asarray(best_gp.x_iters)[:,1]))), 
                        "r--*", linewidth = 1, label=dimensions[1])
        p3, = par2.plot(nums_iter, np.array(list(np.int_(np.asarray(best_gp.x_iters)[:,2]))), 
                        "g--*", linewidth = 1, label=dimensions[2])
        p4, = par3.plot(nums_iter, np.array(list(np.int_(np.asarray(best_gp.x_iters)[:,3]))), 
                        "k--*", linewidth = 1, label=dimensions[3])
        
        axes.set_xlim(0, 15)
        #host.set_ylim(0, 1)
        par1.set_ylim(0, 10)
        par2.set_ylim(1, 128)
        par3.set_ylim(1, 128)
        #iteration during Bayesian optimization\n
        axes.set_xlabel("lon "+str(stat.lon[index])+"°, lat "+str(stat.lat[index])+"°\n"+goodbad[goodbad_ind]+" GPI of "+veg)
        if goodbad_ind == 0:
            axes.set_ylabel(sub_title[var_ind])
#        axes.set_ylabel("RMSE of "+var_name[var_ind])
#        par1.set_ylabel(dimensions[1])
#        par2.set_ylabel(dimensions[2])
#        par3.set_ylabel(dimensions[3])
        
        axes.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        par2.yaxis.label.set_color(p3.get_color())
        par3.yaxis.label.set_color(p4.get_color())
        
        tkw = dict(size=4, width=1.5)
        axes.tick_params(axis='y', colors=p1.get_color(), **tkw)
        par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
        par3.tick_params(axis='y', colors=p4.get_color(), **tkw)
        axes.tick_params(axis='x', **tkw)
        
        i+=1
    lines = [p1, p2, p3, p4]
        
        #host.legend(lines, [l.get_label() for l in lines])
    lgd = fig.legend(lines,     # The line objects [line1, line2, line3, line4]
               labels=[l.get_label() for l in lines],   # The labels for each line
               borderaxespad=0.1,    # Small spacing around legend box
               ncol=4,
               fontsize = smallest_fotsz,
               borderpad=0.5, labelspacing=1.5,
               loc="upper center",
               bbox_to_anchor=(0.55,0.025))
        
    plt.show()
    fig.savefig(save_at+fig_name+'.eps',bbox_extra_artists=(lgd,),bbox_inches = 'tight',format='eps')
    fig.savefig(save_at+fig_name+'.jpeg',bbox_extra_artists=(lgd,),bbox_inches = 'tight',format='jpeg')
save_at = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/IEEEplots/'
var_name = ['sig', 'slope', 'curv']
sub_title= [r'$\sigma_{40}^o$ [dB]',r"$\sigma^{\prime}$ [dB/deg]",r'$\sigma^{\prime\prime}{\rm[dB/deg^2]}$']

GPI = ['Broadleaf', 'Agriculture','Grassland', 'Needleleaf']
veg_class = ["M.C. Broadleaf", "N. Agric.", "M.C. Grasslands", "Les Landes"]

unit = ['[dB]', '[dB/deg]', '[dB/deg]'] # double check the unit for curv!
save_at = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/IEEEplots/'

for j, veg in enumerate(veg_class):
    convergence_plot([GoodIndex_vegClass[j], BadIndex_vegClass[j]], var_name, 
                     veg, unit, 'Fig13_convergence_plot_'+veg, save_at)
#%% Fig 6
# here we can use Manuel's code to plot the figure!

#subPlotMap_general(stat, stat['lon'], stat['lat'],"Fig1_SpatialPattern_ortho.jpeg",save_at+'IEEEplots/',
#           var_stat, sub_title, cb_tick)
def subplot2(ax,fig,img_extent,data_crs, proj,axes_class,i, Z, var,c_label,lon,lat):
    latmax = 52
    lonmin =-5
    lonmax = 10
    latmin = 41#min(lat)
    
    ax.set_extent(img_extent)
    
    coast = cartopy.feature.NaturalEarthFeature(
            category='physical', scale='50m', 
            name='coastline', facecolor='none', edgecolor='k')
    ocean = cartopy.feature.NaturalEarthFeature(
            category='physical', scale='50m', 
            name='ocean', facecolor='#DDDDDD')
   
    ax.add_feature(coast, lw=0.8, alpha=0.5, zorder=9)
    ax.add_feature(ocean, alpha=0.4,zorder=8)
    #ax[0].add_feature(borders, lw=0.4, zorder=2)    
        #viridis_r
    Z = geom_to_masked_cube(Z, country, lat, lon, mask_excludes=True)
    sc = ax.scatter(lon, lat, marker='o', c=Z, transform=data_crs, cmap='YlGnBu',
                    rasterized=True,
                       zorder=7,vmin = 0.1,vmax=0.8, s=4) #zorder decides layer order (1 is bottom)
    if int((i-1)/6) == 0:
        ax.set_title('('+chr(97+i-1)+') '+'{}'.format(c_label),fontdict={'size': smallest_fotsz})
    else:
        ax.set_title('('+chr(97+i-1)+')',fontdict={'size': smallest_fotsz})
        
    fig.canvas.draw()
    yticks = [latmin-2+j*3 for j in range(latmax-latmin)]#[-110, -50, -40, -30, -20, -11, 0, 10, 20, 30, 40, 50]
    xticks = [lonmin-2+j*3 for j in range(lonmax-lonmin)]#[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    ax.gridlines(xlocs=xticks, ylocs=yticks)
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    if i%6 == 1:# or i == 3:
        lambert_yticks(ax, yticks, 'left')
        ax.set_ylabel('Feature Importance \n Fraction [-] ('+var+')',
                            fontsize=smallest_fotsz)
    #if i in [19,20,21,22,23,24]:
    lambert_xticks(ax, xticks, 'bottom')
    # Gridlines
#    gl = ax[0].gridlines(crs=proj, draw_labels=True,
#                  linewidth=1.5, color='black', alpha=0.65, zorder=10,linestyle='--')
#                #alpha sets opacity
#    gl.xlabels_top = False #no xlabels on top (longitudes)
#    gl.ylabels_right = False #no ylabels on right side (latitudes)
#    #gl.xlines = False
#    #gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
#    gl.xformatter = LONGITUDE_FORMATTER
#    gl.yformatter = LATITUDE_FORMATTER
#    gl.xlabel_style = {'size': 8, 'color': 'black', 'weight': 'bold','rotation': 25} #formatting of gridline labels
#    gl.ylabel_style = {'size': 8, 'color': 'black', 'weight': 'bold'}
#    
    return sc


months = ['January', 'Feburary', 'March','April','May','June','July', 'August', 'September','October','November','December']
def PlotMap4_2(Z1,lon, lat,figname,save_at,var,sub_title):
    
    latmax = 52
    lonmin =-5
    lonmax = 10
    latmin = 41#min(lat)
    
    img_extent = [lonmin, lonmax, latmin, latmax]
    
    # Projection system
    data_crs=cartopy.crs.PlateCarree()
    proj=cartopy.crs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=proj))
    
    fig, axes = plt.subplots(nrows=4,ncols=6, #ccrs.PlateCarree()
                             subplot_kw={'projection': cartopy.crs.Orthographic(central_longitude=0,
                                        central_latitude=45, globe=None)} 
                             # https://stackoverflow.com/questions/51621362/subplot-attributeerror-axessubplot-object-has-no-attribute-get-extent
                             )
    #fig = plt.figure()
    fig.set_size_inches(14.5, 12)
    plt.tight_layout(pad = 7, h_pad=3, w_pad=1)
    i = 0
    
    c_label = ['January', 'Feburary', 'March','April','May','June','July', 'August', 'September','October','November','December']
    for ax in axes.flat:
        sc = subplot2(ax,fig,img_extent,data_crs, proj,axes_class,i+1, Z1[(i%6)*2]['{}'.format(var[int(i/6)])], var[int(i/6)],c_label[(i%6)*2],lon,lat)
        i+=1
#    subplot2(fig,img_extent,data_crs, proj,axes_class,2, Z1[1]['{}'.format(var)], var,c_label[1],lon,lat)
#    subplot2(fig,img_extent,data_crs, proj,axes_class,3, Z1[2]['{}'.format(var)], var,c_label[2],lon,lat)
#    subplot2(fig,img_extent,data_crs, proj,axes_class,4, Z1[3]['{}'.format(var)], var,c_label[3],lon,lat)
#    subplot2(fig,img_extent,data_crs, proj,axes_class,5, Z1[4]['{}'.format(var)], var,c_label[4],lon,lat)
#    subplot2(fig,img_extent,data_crs, proj,axes_class,6, Z1[5]['{}'.format(var)], var,c_label[5],lon,lat)
#    subplot2(fig,img_extent,data_crs, proj,axes_class,7, Z1[6]['{}'.format(var)], var,c_label[6],lon,lat)
#    subplot2(fig,img_extent,data_crs, proj,axes_class,8, Z1[7]['{}'.format(var)], var,c_label[7],lon,lat)
#    subplot2(fig,img_extent,data_crs, proj,axes_class,9, Z1[8]['{}'.format(var)], var,c_label[8],lon,lat)
#    subplot2(fig,img_extent,data_crs, proj,axes_class,10, Z1[9]['{}'.format(var)], var,c_label[9],lon,lat)
#    subplot2(fig,img_extent,data_crs, proj,axes_class,11, Z1[10]['{}'.format(var)], var,c_label[10],lon,lat)
#    subplot2(fig,img_extent,data_crs, proj,axes_class,12, Z1[11]['{}'.format(var)], var,c_label[11],lon,lat)
   
    #cax = fig.add_axes([0.95, 0.06, 0.01, 0.88])
    cax = fig.add_axes([0.35, 0.06, 0.3, 0.01])
    cax.tick_params(labelsize=10)
    plt.colorbar(sc,cax=cax, orientation='horizontal')
    cax.set_xlabel('Total feature importance fraction\n{} of year 2017'.format(sub_title),fontdict={'size': smallest_fotsz})
    # Saving figure
#    fig.show()
    plt.savefig('{}{}'.format(save_at,figname+'.jpeg'),bbox_inches = 'tight',format = 'jpeg')
    plt.savefig('{}{}'.format(save_at,figname+'.eps'),bbox_inches = 'tight',format = 'eps')
    
    plt.show()
    plt.close()

def season(shap_time_index,m1):

    season_data= []

    for i in range(len(shap_time_index)):
#        if np.ma.is_masked(france_index[i]) == True:
#            continue
#        print('GPI: '+str(i)+' within France')
        shap_time_index[i].index = pd.to_datetime(shap_time_index[i].index)
        select_data = shap_time_index[i][shap_time_index[i].index.year== 2017] # FOr taking only one year as an example
        select_data = select_data[select_data.index.month == m1]
 
        
        #select_data = shap_time_index[i][shap_time_index[i].index.month == m1] # For averageing over 3 years...
 
        select_data_4 = np.sum(select_data, axis=0)/np.sum(np.sum(select_data, axis=0))

        season_data.append(select_data_4)
    return season_data

def plot_feature_per_month(lon, lat, save_at, fig_name, shap_val_per_gpi_sig, sub_title):
    feature_per_month = []
    
    for i in range(len(months)):
        dataframe = pd.concat(season(shap_val_per_gpi_sig ,i+1), axis=1).T
    #    dataframe['cluster'] = shap_cluster['cluster']
        feature_per_month.append(dataframe)   
    
    
    PlotMap4_2(feature_per_month,lon, lat,fig_name,save_at,input_name, sub_title)
#    PlotMap4_2(feature_per_month,lon, lat,'WG2_all_months',save_at,'WG2','WG2',name, sub_title)
#    PlotMap4_2(feature_per_month,lon, lat,'GPP_all_months',save_at,'GPP','GPP',name, sub_title)
#    PlotMap4_2(feature_per_month,lon, lat,'RE_all_months',save_at,'RE','RE',name, sub_title)
    
    return feature_per_month
#[feature_sig, feature_slope, feature_curv]
save_at = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/IEEEplots/'

sub_title= [r'$\sigma_{40}^o$ [dB]',r"$\sigma^{\prime}$ [dB/deg]",r'$\sigma^{\prime\prime}{\rm[dB/deg^2]}$']

#Z = [feature_sig, feature_slope, feature_curv]
input_name = ['LAI', 'WG2', 'GPP', 'RE']

feature_sig = plot_feature_per_month(df_long.lon, df_long.lat, save_at, 'fig9_Backscatter', df_long.shap_sig, sub_title[0])
feature_slope = plot_feature_per_month(df_long.lon, df_long.lat, save_at,'fig10_slope',df_long.shap_slope, sub_title[1])
feature_curv = plot_feature_per_month(df_long.lon, df_long.lat, save_at,'fig11_curvature',df_long.shap_curv, sub_title[2])

#PlotMap4_2(Z, df_long.lon, df_long.lat,'fig6_monthly averaged shapely',save_at,input_name, sub_title)
#----------------------------------------------------------#
#----------------------------------------------------------#
#----------------------------------------------------------#
#----------------------------------------------------------#  


#%% Fig 9
# plot of hyperparameter
# read the best hyperparameter first!!!
#=============================================================
#=============================================================
#=============================================================
#=============================================================


#%%

Hyper_val_per_gpi = []

#france_index = geom_to_masked_cube(df_long['Vegtype{}'.format(veg_types[i])], country, df_long.lat, df_long.lon, mask_excludes=True)


for i in range(len(df_all_gpi)):
    if os.path.isdir('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_{}/'.format(i)):
        os.chdir('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_{}/'.format(i))
#    if os.path.isfile('C:/Users/manue/OneDrive/Dokumente/Master Thesis - Susan/Publication/DNN Run Jackknife/GPI_{}/Shap_values(2010-2018)_GPI_{}'.format(i,i)):
        file_name = glob.glob('Best_Hyperpara_space_*')
        if os.path.isfile('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_'+str(i)+'/'+str(file_name[0])):
            # Best_Hyperpara_space_2013_GPI_0
            df = open('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_'+str(i)+'/'+str(file_name[0]), 'rb')
            df_all_gpi_s = pickle.load(df)
            df.close()
            df_hyper = df_all_gpi_s[0] #take the min loss
            # https://www.kesci.com/home/project/5cd838110ee9cd002cca9452
            df_hyper = list(df_hyper)
            #print('shap time length: '+str(len(df_shap))+' dfallgpi length: '+str(len(time)))
            
            print(i)        
            
            Hyper_val_per_gpi.append(df_hyper[1])

save_at_hyper = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/'

   
with open('{}df_hyper_xs'.format(save_at_hyper), 'wb') as f:
    pickle.dump([Hyper_val_per_gpi], f)  
#%%
path = "E:/Surfsara Output DNN/DNN Run Output Pickle/"

df = open(path+'France/df_hyper_xs', 'rb')
df_hyper = pickle.load(df)
df.close()
df_hyper = df_hyper[0]

var_hyper = ['learning_rate', 'num_dense_layers', 'num_input_nodes', 
            'num_dense_nodes','activation_method', 'batch_size']

df_hyperPara = pd.DataFrame()
for i in range(len(var_hyper)):
    paraList = []
    for j in range(len(df_hyper)):
        paraList.append(df_hyper[j][i])
    df_hyperPara[var_hyper[i]] = paraList
df_hyperPara['lon'], df_hyperPara['lat'] = stat['lon'], stat['lat']
save_at_hyper = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/'

with open('{}df_hyperPara_df_xs'.format(save_at_hyper), 'wb') as f:
    pickle.dump([df_hyperPara], f)  
#%%
path = "E:/Surfsara Output DNN/DNN Run Output Pickle/"

df = open(path+'France/df_hyperPara_df_xs', 'rb')
df_hyperPara = pickle.load(df)
df.close()
df_hyperPara = df_hyperPara[0]

#%% visulization
# learning_rate, num_dense_layers, num_input_nodes, 
# num_dense_nodes,activation, batch_size
# I think a geo map for those parameters should be fine

def subPlotMap_Hyperpara(Z, lon, lat,figname,save_at,var_stat, sub_title, cb_tick): 
    latmax = 52
    lonmin =-5
    lonmax = 10
    latmin = 41#min(lat)
    
    img_extent = [lonmin, lonmax, latmin, latmax]
    
    # Projection system
    data_crs=cartopy.crs.PlateCarree()
    proj=cartopy.crs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=proj))
    
    # Colorbar limits
    #cbmax = 6
    #cbmin = 1
    
    # Data and axes
    #x, y = np.meshgrid(lon,lat)
    #fig = plt.figure() #1, (4., 4.)
    #ax_pcs = fig.add_subplot(2,3,i+1)
#    fig.set_size_inches(7.25, 6)

#    ax = AxesGrid(fig, 111, axes_class=axes_class,
#                  axes_pad=0.3,  # pad between axes in inch.
#              nrows_ncols=(2, 3), label_mode='')
    coast = cartopy.feature.NaturalEarthFeature(
            category='physical', scale='50m', 
            name='coastline', facecolor='none', edgecolor='k')
    #    land = cartopy.feature.NaturalEarthFeature(
    #            category='physical', scale='10m', name='coastline', facecolor='none')
    ocean = cartopy.feature.NaturalEarthFeature(
            category='physical', scale='50m', 
            name='ocean', facecolor='#DDDDDD')
    #    borders = cartopy.feature.NaturalEarthFeature(
    #            category='cultural', scale='10m',
    #            name='admin_0_boundary_lines_land', facecolor='none', edgecolor='black')
    #projection = ccrs.epsg(32636)
    fig, axes = plt.subplots(nrows=2,ncols=3,
                             subplot_kw={'projection': ccrs.PlateCarree()} 
                             # https://stackoverflow.com/questions/51621362/subplot-attributeerror-axessubplot-object-has-no-attribute-get-extent
                             )
    fig.set_size_inches(10, 6)
    plt.tight_layout(pad = 3.28, h_pad=2.08, w_pad=6)
    i = 0
    for ax in axes.flat:
        #ax = fig.add_subplot(2,3,i+1)
        #        ax = AxesGrid(fig, 231+i, axes_class=axes_class,
#                  axes_pad=0.5,  # pad between axes in inch.
#              nrows_ncols=(1, 1), label_mode='')
        # if this method is used, do not add the crrs in plt.subplots!
#
        ax.set_extent(img_extent)
        #ax[0].add_feature(land, lw=0.4, alpha=0.7, zorder=1)
        ax.add_feature(coast, lw=0.8, alpha=0.5, zorder=9)
        ax.add_feature(ocean, alpha=0.4,zorder=8)
        
        if i == 4:
            Z[var_stat[i]].replace({'tanh': 0, 'relu': 1}, inplace = True)
            # or df.replace(['C', 'F'], [0.999, 0.777]) x
        #temp = pd.DataFrame(Z[var_stat[i]])
        
        temp = geom_to_masked_cube(Z[var_stat[i]], country, lat, lon, mask_excludes=True)
        if i >= 1:
            cmap = plt.get_cmap('RdYlGn', np.max(temp)-np.min(temp)+1)
        elif i == 4:
            cmap = plt.get_cmap('RdYlGn', 2)
        else:
            cmap = 'RdYlGn'
        sc = ax.scatter(lon, lat, marker='o', c=temp, transform=data_crs, cmap=cmap,
                        zorder=7,
                       #vmin=0.75,vmax=0.9, 
                        s=5) #zorder decides layer order (1 is bottom)
        ax.set_title(label = '('+chr(97+i)+') '+sub_title[i], fontsize = smallest_fotsz, loc = 'center')
        
        # Gridlines
        gl = ax.gridlines(crs=proj, draw_labels=True,
                      linewidth=1.5, color='black', alpha=0.65, zorder=10,linestyle='--')
                    #alpha sets opacity
        gl.xlabels_top = False #no xlabels on top (longitudes)
        gl.ylabels_right = False #no ylabels on right side (latitudes)
        if i != 0 and i != 3:
            gl.ylabels_left = False
        
        axpos=ax.get_position()
        cbar_ax=fig.add_axes([axpos.x1, axpos.y0+0.03,0.0075, axpos.height-0.055]) #l, b, w, h
#            divider = make_axes_locatable(ax[0])
#            cax = divider.append_axes("right", size="5%", pad=0.05)
#
#            cbar=fig.colorbar(sc, cax=cax)
        cbar=fig.colorbar(sc,cax=cbar_ax)
        if i == 4:
            cbar.set_ticks([0.25,0.75])
            cbar.set_ticklabels(['tanh','relu'])
        cbar.ax.tick_params(labelsize=smallest_fotsz, which='both')
#        if i == 2 or i == 5:
#            cbar.set_label(cb_tick[int(i/3)], fontsize = smallest_fotsz)#'Pearson correlation coefficient')
        
        gl.xlocator = xmajorLocator
        gl.ylocator = ymajorLocator
        ax.xaxis.set_minor_locator(xminorLocator)  
        ax.yaxis.set_minor_locator(yminorLocator)  
        
        ax.xaxis.grid(True, which='major', linestyle='--') #x - major  
        ax.yaxis.grid(True, which='minor', linestyle='--') #y - major
        
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': smallest_fotsz, 'color': 'black',
#                           'weight': 'bold',
                           'rotation': 45} #formatting of gridline labels
        gl.ylabel_style = {'size': smallest_fotsz, 
#                           'weight': 'bold',
                           'color': 'black'}
        
        make_axes_locatable(ax) #divider = 
        i = i+1
    fig.show()
    plt.savefig('{}{}'.format(save_at,figname),bbox_inches = 'tight',format='jpeg')
    # Colorbar
var_hyper = ['learning_rate', 'num_dense_layers', 'num_input_nodes', 
            'num_dense_nodes','activation_method', 'batch_size']
cb_tick = ['mean', 'standard deviation']
sub_title= ['learning rate', 'num dense layers', 'num input nodes', 
            'num dense nodes','activation method', 'batch size']

    
subPlotMap_Hyperpara(df_hyperPara, df_hyperPara['lon'], df_hyperPara['lat'],"Fig9_hyperPara.jpeg",save_at+'IEEEplots/',
           var_hyper, sub_title, cb_tick)

#%%
df = open('E:/Surfsara Output DNN/DNN Run Output Pickle/France/df_stat_veg_manuel', 'rb')
clusters = pickle.load(df)
df.close()
stat_veg  = clusters [0]
#%%
# Fig 7 and Fig 8
# of which the smallest and largest RMSE will be chosen?
# maybe we don't need this figure???
def timeseries_BestvsWorst(Z, ylim, var_name, cluster_name, unit, fig_name, save_at):

    # using all of the variables to plot a 3*2 subuplot
    # var_name (3) x cluster_name (2) (good vs bad)
    
    fig, ax = plt.subplots(nrows=3,ncols=2, sharey='row')
    fig.set_size_inches(28, 12)
    #plt.tight_layout(pad = 1.08, h_pad=1.08, w_pad=None)
    i = 0
    for axes in ax.flat:
        #cluster_ind = i%4
        var_ind = int(i/2)
        goodbad_ind = int(i%2)
        goodbad = ['Good', 'bad']
        
        line_list = []
        line1_list= []
        line2_list= []
        for input_ind in range(len(input_name)):
            line = axes.plot(Z[goodbad_ind][var_ind]['data']['predicted_{}'.format(var_name[var_ind])], 
#                             sharey='row',
                      color='grey', linewidth=1)
            line1 = axes.plot(Z[goodbad_ind][var_ind]['data']['observed_{}'.format(var_name[var_ind])], 
                              linewidth=0.8,
#                              sharey='row',
                      color='r', linestyle=':')
            line2 = axes.plot(Z[goodbad_ind][var_ind]['data']['predicted_{}'.format(var_name[var_ind])][Z[goodbad_ind][var_ind]['data']['predicted_{}'.format(var_name[var_ind])].index.year >= 2017], 
#                              sharey='row',
                      color='blue')
            line_list.append(line)
            line1_list.append(line1)
            line2_list.append(line2)
        
        if i == 0 or i == 2 or i == 4:
            axes.set_ylabel(yaxis_label[var_ind],
                            fontsize=small_fontsize)
        # sigma: [5,15]
        # slope: [-0.22, -0.015]
        # curv: [-0.004, 0.006]
        #axes.set_ylim(ylim[var_ind])
        
        temp = Z[goodbad_ind][var_ind]['data']['predicted_{}'.format(var_name[var_ind])]
        temp.index = pd.to_datetime(temp.index)
        axes.xaxis.set_major_locator(xmajorLocator) 
        
        for tick in axes.xaxis.get_major_ticks():  
            tick.label1.set_fontsize(small_fontsize)
        for tick in axes.yaxis.get_major_ticks():  
            tick.label1.set_fontsize(small_fontsize)
        
        rmse = sqrt(mean_squared_error(Z[goodbad_ind][var_ind]['data']['predicted_{}'.format(var_name[var_ind])],
                    Z[goodbad_ind][var_ind]['data']['observed_{}'.format(var_name[var_ind])]))
#        cluster_name = df_long[df_long.lon == Z[goodbad_ind][var_ind]['data'].lon[0]][df_long.lat == Z[goodbad_ind][var_ind]['data'].lat[0]].cluster.values[0]
#        
        if var_ind == 0:
            axes.set_title(#goodbad[goodbad_ind]+'example\n'
                           '('+chr(97+i)+') {}, {}°N, {}°E ({}), RMSE = {}'.format(
                           yaxis_label[var_ind],
                           Z[goodbad_ind][var_ind]['data'].lat[0],
                           Z[goodbad_ind][var_ind]['data'].lon[0],
                           cluster_name,
                           round(rmse,4)),
                           fontsize = small_fontsize)
        else:
            axes.set_title('('+chr(97+i)+') {}, {}°N, {}°E ({}), RMSE = {}'.format(
                           yaxis_label[var_ind],
                           Z[goodbad_ind][var_ind]['data'].lat[0],
                           Z[goodbad_ind][var_ind]['data'].lon[0],
                           cluster_name,
                           round(rmse,4)),
                           fontsize = small_fontsize)
        # add legend outside the subplots
        #lg_ax = fig.add_axes([0.15, 0.15, 0.7, 0.05])
        
        i = i+1
    #h,l = axes.get_legend_handles_labels()
    lgd = fig.legend([line_list[0], line1_list[0], line2_list[0]],     # The line objects [line1, line2, line3, line4]
               labels=legend_name,   # The labels for each line
               borderaxespad=0.1,    # Small spacing around legend box
               ncol=4,
               fontsize = small_fontsize,
               borderpad=1.5, labelspacing=1.5,
               loc="upper center",
               bbox_to_anchor=(0.5-0.07/2,0.1-0.01))
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.subplots_adjust(left=0.07, right=0.93, wspace=0.15, hspace=0.35,
                        bottom = 0.15)
    plt.savefig('{}{}'.format(save_at,fig_name+'.eps'),bbox_extra_artists=(lgd,), bbox_inches='tight', format = 'eps')
    plt.savefig('{}{}'.format(save_at,fig_name+'.jpeg'),bbox_extra_artists=(lgd,), bbox_inches='tight', format = 'jpeg')
    
    plt.show()
    #plt.close()

xmajorLocator   = MultipleLocator(2*365)
ymajorLocator   = MultipleLocator(2)
xminorLocator   = MultipleLocator(1)
yminorLocator   = MultipleLocator(1)

veg_name = ['Vegtype4', 'Vegtype7', 'Vegtype10', 'Vegtype15']
#veg_class = ['broadleaf', 'agriculture', 'grass', 'needleleaf']
veg_class = ["M.C. Broadleaf", "N. Agric.", "M.C. Grasslands", "Les Landes"]

GoodIndex_vegClass, BadIndex_vegClass = [],[]
GoodGPI_vegClass, BadGPI_vegClass = [],[]
for j, veg in enumerate(veg_class):
    GoodIndex, BadIndex = [],[]
    GoodGPI, BadGPI = [],[]
    #stat_data = [stat.RMSE_sig, stat.RMSE_slope, stat.RMSE_curv]
    stat_data = [stat_veg.RMSE_sig, stat_veg.RMSE_slope, stat_veg.RMSE_curv]
    for i in range(3):
#        temp = geom_to_masked_cube(stat_data[i], country, df_long.lat, df_long.lon, mask_excludes=True)
        temp = stat_data[i][j]
        GoodIndex.append(temp.argmin())
        BadIndex.append(temp.argmax())
        GoodGPI.append(df_long.iloc[GoodIndex[i]])
        BadGPI.append(df_long.iloc[BadIndex[i]])
    GoodGPI_vegClass.append(GoodGPI)
    BadGPI_vegClass.append(BadGPI)
    GoodIndex_vegClass.append(GoodIndex)
    BadIndex_vegClass.append(BadIndex)
ylim = []
for j, veg in enumerate(veg_class):
    ylim_veg = []
    for i in range(3):
        ylim_var = []
        upper1 = np.max(np.max(GoodGPI_vegClass[j][i]['data'][['predicted_{}'.format(var_name[i]), 'observed_{}'.format(var_name[i])]]))
        lower1 = np.min(np.min(GoodGPI_vegClass[j][i]['data'][['predicted_{}'.format(var_name[i]), 'observed_{}'.format(var_name[i])]]))
        upper2 = np.max(np.max(BadGPI_vegClass[j][i]['data'][['predicted_{}'.format(var_name[i]), 'observed_{}'.format(var_name[i])]]))
        lower2 = np.min(np.min(BadGPI_vegClass[j][i]['data'][['predicted_{}'.format(var_name[i]), 'observed_{}'.format(var_name[i])]]))
        #ylim_veg.append([np.max([upper1, upper2]), np.min([lower1, lower2])])
        ylim_veg.append([np.min([lower1, lower2]), np.max([upper1, upper2])])
    
    ylim.append(ylim_veg)

input_name = ['LAI','WG2','GPP','RE']
var_name = ['sig', 'slope', 'curv']
legend_name = ['prediction', 'observation', 'validation']
yaxis_label= [r'$\sigma_{40}^o$ [dB]',r"$\sigma^{\prime}$ [dB/deg]",r'$\sigma^{\prime\prime}{\rm[dB/deg^2]}$']
#unit = ['[dB]', '[dB/deg]', '[dB/deg]'] # double check the unit for curv!
save_at_fig7 = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/IEEEplots/'

#timeseries_BestvsWorst([GoodGPI, BadGPI], var_name, yaxis_label,'fig7_GoodvsBad',save_at_fig7)
for j, veg in enumerate(veg_class):
    timeseries_BestvsWorst([GoodGPI_vegClass[j], BadGPI_vegClass[j]], ylim[j], var_name, veg, yaxis_label,'fig7_GoodvsBad'+'_'+veg, save_at_fig7)

#%%
#pick up some representative GPIs (iteration process)
bestmodel_goodGPI_vegClass, bestmodel_badGPI_vegClass = [], []

for j in range(len(GoodIndex_vegClass)):
    temp = []
    for i in GoodIndex_vegClass[j]:
        if os.path.isdir('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_{}/'.format(i)):
            os.chdir('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_{}/'.format(i))
            file_name = glob('best_opt_model_*')
            
            if os.path.isfile('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_'+str(i)+'/'+str(file_name[0])):
                df = load_model('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_'+str(i)+'/'+str(file_name[0]))
                temp.append(df)
    bestmodel_goodGPI_vegClass.append(temp)
for j in range(len(BadIndex_vegClass)):
    temp = []
    for i in BadIndex_vegClass[j]:
        if os.path.isdir('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_{}/'.format(i)):
            os.chdir('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_{}/'.format(i))
            file_name = glob('best_opt_model_*')
            
            if os.path.isfile('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_'+str(i)+'/'+str(file_name[0])):
                df = load_model('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_'+str(i)+'/'+str(file_name[0]))
                temp.append(df)
    bestmodel_badGPI_vegClass.append(temp)

df = open('{}Hyperparameter_space_{}_{}'.format(new_folder,rmse_min_year,GPI), 'rb')
best_hyper = pickle.load(df)
df.close()
best_hyper = best_hyper[0]

#%%
"""
bestmodel_val_per_gpi = []

#france_index = geom_to_masked_cube(df_long['Vegtype{}'.format(veg_types[i])], country, df_long.lat, df_long.lon, mask_excludes=True)


for i in range(len(df_all_gpi)):
    if os.path.isdir('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_{}/'.format(i)):
        os.chdir('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_{}/'.format(i))
#    if os.path.isfile('C:/Users/manue/OneDrive/Dokumente/Master Thesis - Susan/Publication/DNN Run Jackknife/GPI_{}/Shap_values(2010-2018)_GPI_{}'.format(i,i)):
        file_name = glob('best_opt_model_*')
        if os.path.isfile('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_'+str(i)+'/'+str(file_name[0])):
            # Best_Hyperpara_space_2013_GPI_0
            df = load_model('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_'+str(i)+'/'+str(file_name[0]))
            
            #df = open('E:/Surfsara Output DNN/DNN Run Jackknife/GPI_'+str(i)+'/'+str(file_name[0]), 'rb')
            df_all_gpi_s = pickle.load(df)
            df.close()
            df_bestmodel = df_all_gpi_s[0] #take the min loss
            # https://www.kesci.com/home/project/5cd838110ee9cd002cca9452
            df_bestmodel = list(df_bestmodel)
            #print('shap time length: '+str(len(df_shap))+' dfallgpi length: '+str(len(time)))
            
            print(i)        
            
            bestmodel_val_per_gpi.append(df_bestmodel[1])

save_at_bestmodel = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/'

   
with open('{}df_bestmodel_xs'.format(save_at_bestmodel), 'wb') as f:
    pickle.dump([bestmodel_val_per_gpi], f)
    """
#%%
max_types_series = []
precent_veg = []
veg_types = [4,7,10,15]
for i in range(len(veg_types)):
    france_index = geom_to_masked_cube(df_long['Vegtype{}'.format(veg_types[i])], country, df_long.lat, df_long.lon, mask_excludes=True)


#    stored = df_long[df_long['Vegtype{}'.format(veg_types[i])]==np.max(df_long['Vegtype{}'.format(veg_types[i])])]
    stored = df_long[df_long['Vegtype{}'.format(veg_types[i])]==np.max(france_index)]
    
    
    #stored = df_long[df_long['Vegtype{}'.format(veg_types[i])]==np.max(df_long['Vegtype{}'.format(veg_types[i])])]
    max_types_series.append(stored)
    print(stored.index[0])
# 667 924 871 163
#%%

# sensitivity towards parameters
# calculate the jacobian mateix?
# https://medium.com/unit8-machine-learning-publication/computing-the-jacobian-matrix-of-a-neural-network-in-python-4f162e5db180

def Jacob_output_paras(save_at, agri_all, k):
    agri_all.index = pd.to_datetime(agri_all.index.values)
     
    df_all_gpi_norm = pd.DataFrame()
    

    pre_Method_LAI = StandardScaler()
    pre_Method_WG2 = StandardScaler()
    pre_Method_GPP = StandardScaler()
    pre_Method_RE = StandardScaler()
    
    pre_Method_sig = StandardScaler()
    pre_Method_slope = StandardScaler()
    pre_Method_curv = StandardScaler()
    
    # normalization did by PCA!!!
    df_all_gpi_norm['LAI'] = pre_normalization(agri_all['Input_LAI'], pre_Method_LAI)
    df_all_gpi_norm['WG2'] = pre_normalization(agri_all['Input_WG2'], pre_Method_WG2)
    df_all_gpi_norm['GPP'] = pre_normalization(agri_all['Input_GPP'], pre_Method_GPP)
    df_all_gpi_norm['RE'] = pre_normalization(agri_all['Input_RE'], pre_Method_RE)

    
    df_all_gpi_norm['sig'] = pre_normalization(agri_all['observed_sig'], pre_Method_sig)
    df_all_gpi_norm['slope'] = pre_normalization(agri_all['observed_slope'], pre_Method_slope)
    df_all_gpi_norm['curv'] = pre_normalization(agri_all['observed_curv'], pre_Method_curv)
   
    df_all_gpi_norm['lon'] = agri_all['lon'].values
    df_all_gpi_norm['lat'] = agri_all['lat'].values
      
    df_all_gpi_norm.index = agri_all.index
    
    val_split_year = 2017
    #jackknife_all = df_all_gpi_norm[df_all_gpi_norm.index.year < val_split_year]
    vali_all = df_all_gpi_norm[df_all_gpi_norm.index.year >= val_split_year]
    
    lables_val, input_val = get_data_labels(vali_all)
    # import the best model
    
    # remember that k is the index of GPI!!!!
    # learning_rate, num_dense_layers,num_input_nodes,
                 #num_dense_nodes, activation
#    var_hyper = ['learning_rate', 
#                 'num_dense_layers', 'num_input_nodes', 
#            'num_dense_nodes','activation_method', 'batch_size']
    best_model = keras_dnn(df_hyperPara.iloc[k]['learning_rate'],
                           df_hyperPara.iloc[k]['num_dense_layers'],
                           df_hyperPara.iloc[k]['num_input_nodes'],
                           df_hyperPara.iloc[k]['num_dense_nodes'],
                           df_hyperPara.iloc[k]['activation_method']
                           )
    [learning_rate, num_dense_layers,num_input_nodes,
        num_dense_nodes, activation] = [df_hyperPara.iloc[k]['learning_rate'],
                                       df_hyperPara.iloc[k]['num_dense_layers'],
                                       df_hyperPara.iloc[k]['num_input_nodes'],
                                       df_hyperPara.iloc[k]['num_dense_nodes'],
                                       df_hyperPara.iloc[k]['activation_method']]
    var_hyper_total = [learning_rate, num_dense_layers,num_input_nodes,
                       num_dense_nodes, activation]
    
    epsilon = [10,128,128]
    predicted = best_model.predict(input_val)
    #predicted = agri_all[['predicted_sig','predicted_slope', 'predicted_curv']]
#    j = 0
    jacob_sig, jacob_slop, jacob_curv = [],[],[]#pd.DataFrame()
    new_metrics_perGPI = []
    for n, x in enumerate(var_hyper_total):
        print(n)
        # num_dense_layers num_input_nodes num_dense_nodes
        # activation_method batch_size
        var_hyper_total = [learning_rate, num_dense_layers,num_input_nodes,
                       num_dense_nodes, activation]
        
        if x in [learning_rate]:
            pass
        
        if x in [activation]:
            if x  == 'tanh':
                var_hyper_total[n] = 'relu'
            else:
                var_hyper_total[n] = 'tanh'
                
        if n in [1, 2, 3]:
            new_metrics = np.zeros([3,epsilon[n-1],2]) 
            # num of var, num of pertubations
            for m in range(1,epsilon[n-1]):
                var_hyper_total[n] = m # should not be += m!!!!! should just change it to m!!!
        
                best_model_change = keras_dnn(var_hyper_total[0],
                                      var_hyper_total[1],
                                      var_hyper_total[2],
                                      var_hyper_total[3],
                                      var_hyper_total[4]
                                      )
                predicted_change = best_model_change.predict(input_val)
                for l in range(3):
                    if l == 0: 
                        pre_Method = pre_Method_sig
                    if l == 1:
                        pre_Method = pre_Method_slope
                    if l == 2:
                        pre_Method = pre_Method_curv
                    re_predicted = np.asarray(pre_Method.inverse_transform(predicted_change[:,l].reshape(-1,1)), 'f')
                    re_label = np.asarray(pre_Method.inverse_transform(lables_val[:,l].reshape(-1,1)), 'f')
                    
                    # rmse, mae, pearson, spearson
                    new_metrics[l,m, 0] = np.round(np.sqrt(((np.concatenate(re_predicted) - np.concatenate(re_label)) ** 2).mean()),5)
                    #new_metrics[l,m] = np.round(np.sqrt((np.abs(np.concatenate(re_predicted) - np.concatenate(re_label))).mean()),5)
                    pearsCor_t = pearsonr(np.asarray(re_predicted,dtype='float').reshape([-1,]), np.asarray(re_label,dtype='float').reshape([-1,]))
                    
                    new_metrics[l,m, 1] = pearsCor_t[0]
            print('===============completed hyperpara'+str(n)+' =============')
            new_metrics_perGPI.append(new_metrics)
#                dif = predicted - predicted_change
#                jacob_sig.append(dif[:,0].reshape(-1,1)) # 3 output!!!!
#                jacob_slop.append(dif[:,1].reshape(-1,1)) # 3 output!!!!
#                jacob_curv.append(dif[:,2].reshape(-1,1)) # 3 output!!!!
    return new_metrics_perGPI#jacob_sig, jacob_slop, jacob_curv

save_at = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/IEEEplots/'
var_hyper_iter = [#'learning_rate', 
                 'num_dense_layers', 'num_input_nodes', 
            'num_dense_nodes','activation_method']#, 'batch_size']
#jacob_sig_mean, jacob_slop_mean, jacob_curv_mean = [],[],[]
new_metrics_list = []
for i in range(len(max_types_series)):
    new_metrics_perGPI = Jacob_output_paras(save_at, max_types_series[i]['data'].iloc[0], i)
    new_metrics_list.append(new_metrics_perGPI)
#    jacob_sig_mean.append(jacob_sig)
#    jacob_slop_mean.append(jacob_slop)
#    jacob_curv_mean.append(jacob_curv)
    print(i)
#df_jacob = pd.DataFrame()
#df_jacob['lat'], df_jacob['lon'] = df_long.lat, df_long.lon
#df_jacob.index = df_long.index
#df_jacob['sig'], df_jacob['slop'], df_jacob['curv'] = jacob_sig_mean, jacob_slop_mean, jacob_curv_mean
#vali_all = df_all_gpi_norm[df_all_gpi_norm.index.year >= val_split_year]
#%%
# save the file

path_newmetrics = "E:/Surfsara Output DNN/DNN Run Output Pickle/France"
with open(path_newmetrics+'new_metrics_list', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(new_metrics_list, filehandle)
# how to plot it?
#%%
path_newmetrics = "E:/Surfsara Output DNN/DNN Run Output Pickle/France"
df = open(path_newmetrics+'new_metrics_list', 'rb')
new_metrics_file = pickle.load(df)
df.close()
#new_metrics_file = new_metrics_file[0]
#%% plot the sensitivity of parameters
def paraSensitivity(Z, yaxis_label, fig_name, save_at, k):
    # new metrics files: 4, 3, (3, 10, 2)
    # GPIs, paras, (sig/slop/curv, pertubation, pearson/RMSE) of the new model
    # should be 3 plots for sig/slop/curv
    # for each plot, 4 columns for 4 GPIs, and 3 rows for 3 paras
    # using all of the variables to plot a 3*2 subuplot
    # var_name (3) x cluster_name (2) (good vs bad)
    
    fig, ax = plt.subplots(nrows=3,ncols=4)
    fig.set_size_inches(30, 10)
    #plt.tight_layout(pad = 1.08, h_pad=1.08, w_pad=None)
    i = 0
    line_list = []
    line1_list= []
    for axes in ax.flat:
        #cluster_ind = i%4
        GPI_ind = int(i%4)
        #var_ind = int(i/2)
        para_ind = int(i/4)
#        goodbad_ind = int(i%2)
#        goodbad = ['Good', 'bad']
        
        #for input_ind in range(2):
        line = axes.plot(Z[GPI_ind][para_ind][k,:,0], 
                         label = 'RMSE',
                      color='grey', linewidth=1) # rmse
        line2 = axes.plot([], [], ':r', label = 'Pearson Correlation Coefficient')
        axes.tick_params(axis='y', labelcolor='grey')
        line_list.append(line)
        
        ax2 = axes.twinx()
        line1= ax2.plot(Z[GPI_ind][para_ind][k,:,1], 
                        label = 'Pearson Correlation Coefficient',
                         linestyle=':',
                         color='r', linewidth=1)
        ax2.tick_params(axis='y', labelcolor='r')
        line1_list.append(line1)
        if i in [0,4,8]:
            axes.set_ylabel(yaxis_label[para_ind],
                            fontsize=small_fontsize)
        if para_ind == 2:
            axes.set_xlabel('pertubation' , fontsize=small_fontsize)
        
        cluster_name = df_long[df_long.lon == max_types_series[GPI_ind].lon.values[0]][df_long.lat == max_types_series[GPI_ind].lat.values[0]].cluster.values[0]
       
        if para_ind == 0:
            axes.set_title(#goodbad[goodbad_ind]+'example\n'
                           '('+chr(97+i)+') {}, {}°N {}°E (cluster {})'.format(
                           yaxis_label[para_ind],
                           max_types_series[GPI_ind].lat.values[0],
                           max_types_series[GPI_ind].lon.values[0],
                           cluster_name),
                           fontsize = small_fontsize)
        else:
            axes.set_title('('+chr(97+i)+') {}, {}°N {}°E (cluster {})'.format(
                           yaxis_label[para_ind],
                           max_types_series[GPI_ind].lat.values[0],
                           max_types_series[GPI_ind].lon.values[0],
                           cluster_name),
                           fontsize = small_fontsize)        
        i = i+1
    lines, labels = axes.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #print(labels+labels2)
    lgd = fig.legend(line+line2, # The line objects [line1, line2, line3, line4]
               labels=legend_name,   # The labels for each line
               borderaxespad=0.1,    # Small spacing around legend box
               ncol=4,
               borderpad=1.5, labelspacing=1.5,
               loc="upper center",
               bbox_to_anchor=(0.5-0.07/2,0.1-0.01))
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.subplots_adjust(left=0.07, right=0.93, wspace=0.3, hspace=0.35,
                        bottom = 0.15)
    plt.savefig('{}{}'.format(save_at,fig_name),bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    #plt.close()

input_name = ['LAI','WG2','GPP','RE']
var_name = ['sig', 'slope', 'curv']
legend_name = ['RMSE', 'Pearson Coefficient']
#num_dense_layers num_input_nodes num_dense_nodes
yaxis_label = ['num of layers', 'num of input nodes', 'num of dense nodes']
#yaxis_label= [r'$\sigma_{40}^o$ [dB]',r"$\sigma^{\prime}$ [dB/deg]",r'$\sigma^{\prime\prime}{\rm[dB/deg^2]}$']
save_at_fig10 = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/IEEEplots/'

for k in range(3):
    paraSensitivity(new_metrics_file, yaxis_label,
                           'fig10_SensOfPara'+var_name[k],save_at_fig10, k)

#%% sensitivity to the local input pertubation!!!!
# normalized sensitivity coefficient, NSC
# +- 5%!!!!
# should be one subplots of rows and cols
# one map: 
def Jacob_localinput(save_at, agri_all, k):
    agri_all.index = pd.to_datetime(agri_all.index.values)
    df_all_gpi_norm = pd.DataFrame()

    pre_Method_LAI = StandardScaler()
    pre_Method_WG2 = StandardScaler()
    pre_Method_GPP = StandardScaler()
    pre_Method_RE = StandardScaler()
    
    pre_Method_sig = StandardScaler()
    pre_Method_slope = StandardScaler()
    pre_Method_curv = StandardScaler()
    
    df_all_gpi_norm['LAI'] = pre_normalization(agri_all['LAI'], pre_Method_LAI)
    df_all_gpi_norm['WG2'] = pre_normalization(agri_all['WG2'], pre_Method_WG2)
    df_all_gpi_norm['GPP'] = pre_normalization(agri_all['GPP'], pre_Method_GPP)
    df_all_gpi_norm['RE'] = pre_normalization(agri_all['RE'], pre_Method_RE)
    
    df_all_gpi_norm['sig'] = pre_normalization(agri_all['sig'], pre_Method_sig)
    df_all_gpi_norm['slope'] = pre_normalization(agri_all['slop'], pre_Method_slope)
    df_all_gpi_norm['curv'] = pre_normalization(agri_all['curv'], pre_Method_curv)
   
    df_all_gpi_norm['lon'] = agri_all['lon'].values
    df_all_gpi_norm['lat'] = agri_all['lat'].values
      
    df_all_gpi_norm.index = agri_all.index
    
    val_split_year = 2017
    #jackknife_all = df_all_gpi_norm[df_all_gpi_norm.index.year < val_split_year]
    vali_all = df_all_gpi_norm#[df_all_gpi_norm.index.year >= val_split_year]
    
    lables_val, input_val = get_data_labels(vali_all)
    # import the best model
    
    # remember that k is the index of GPI!!!!
    mypath = 'E:/Surfsara Output DNN/DNN Run Jackknife/GPI_{}/'.format(k)
        
    x =[os.path.join(root, f) for root, _, files in os.walk(mypath)
                               for f in files
                               if f.startswith('best_opt_model')][0]
        
    best_model =  load_model('{}'.format(x))
#    best_model = keras_dnn(df_hyperPara.iloc[k]['learning_rate'],
#                           df_hyperPara.iloc[k]['num_dense_layers'],
#                           df_hyperPara.iloc[k]['num_input_nodes'],
#                           df_hyperPara.iloc[k]['num_dense_nodes'],
#                           df_hyperPara.iloc[k]['activation_method']
#                           )
    input_name = ['LAI', 'WG2', 'GPP', 'RE']
    #epsilon = [10,128,128]
    predicted = best_model.predict(input_val)
    #predicted = agri_all[['predicted_sig','predicted_slope', 'predicted_curv']]
#    j = 0
    #jacob_sig, jacob_slop, jacob_curv = [],[],[]#pd.DataFrame()
    NSC_perGPI = []
#    np.random.seed(0)
    df_NSC_perGPI = pd.DataFrame()
    for x in range(len(input_name)):
        # LAI WG2 GPP RE
#        print(x)
#        print(input_val.shape)
#        print(input_val[x,:].shape[1])
#        s = np.random.normal(3, 1, (1,input_val[:,x].shape[0]))
#        input_val[:,x] = s+input_val[:,x]
        input_val[:,x] = 1.05*input_val[:,x]
        
        predicted_change = best_model.predict(input_val)
        output = ['sig', 'slope', 'curv']
        for l in range(3):
            if l == 0: 
                pre_Method = pre_Method_sig
            if l == 1:
                pre_Method = pre_Method_slope
            if l == 2:
                pre_Method = pre_Method_curv
            re_predicted = np.asarray(pre_Method.inverse_transform(predicted[:,l].reshape(-1,1)), 'f')
            re_predicted_change = np.asarray(pre_Method.inverse_transform(predicted_change[:,l].reshape(-1,1)), 'f')
            
            NSC = (re_predicted_change - re_predicted)/re_predicted/0.05 #*input_val[x,:].reshape(-1,1)/s
            #re_label = np.asarray(pre_Method.inverse_transform(lables_val[:,l].reshape(-1,1)), 'f')
            
            print('===============completed pertubation of input'+str(input_name[x])+' of GPI'+str(k)+' =============')
            
            NSC_perGPI.append(NSC)
#                dif = predicted - predicted_change
#                jacob_sig.append(dif[:,0].reshape(-1,1)) # 3 output!!!!
#                jacob_slop.append(dif[:,1].reshape(-1,1)) # 3 output!!!!
#                jacob_curv.append(dif[:,2].reshape(-1,1)) # 3 output!!!!
            df_NSC_perGPI[input_name[x]+'_'+output[l]] = list(NSC.reshape(-1,))
            # vali_all
    #df_NSC_perGPI['lon'] = agri_all['lon']
    #df_NSC_perGPI['lat'] = agri_all['lat']
    return df_NSC_perGPI#NSC_perGPI#jacob_sig, jacob_slop, jacob_curv

save_at = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/IEEEplots/'

NSC_list = []
#for i in range(len(df_all_gpi)):
index_list = []
maxvegindex_list = [667, 924, 871, 163]
for j in range(4):
    for i in range(len(df_all_gpi)):
        if df_all_gpi[i].empty:
            print('Empty')
        elif df_all_gpi[i].lat.values[0] == max_types_series[j].lat.values[0] and df_all_gpi[i].lon.values[0] == max_types_series[j].lon.values[0]:
            index_list.append(i)
    
    
    
for i in index_list: #[671, 930, 877, 163]
    one_gpi = df_all_gpi[i]
    if one_gpi.empty:
        print('Empty')
    else:
        NSC_perGPI = Jacob_localinput(save_at, one_gpi, i)
        NSC_list.append(NSC_perGPI)
        print(i)
#%%
# plot of sensitivity to local inputs
def plotting_time_types(NSC_list,
                        pick_year,
                        input_name, var_name, fig_name, 
                        unit, GPI, save_at):
    # using all of the variables to plot a 3*4 subuplot
    # var_name (3) x cluster_name (4)
    
    fig, ax = plt.subplots(nrows=3,ncols=4)
    fig.set_size_inches(28, 12)
    #plt.tight_layout(pad = 1.08, h_pad=1.08, w_pad=None)
    i = 0
    for axes in ax.flat:
        cluster_ind = i%4
        var_ind = int(i/4)
        lon = max_types_series[cluster_ind]['data'].iloc[0]['lon'][0]
        lat = max_types_series[cluster_ind]['data'].iloc[0]['lat'][0]
        line_list = []
#        im = ax_pcs.imshow(temp.resample('M').mean().dropna(axis=0,how='any').transpose().values,
#                           vmin=0, vmax=1)
        for input_ind in range(len(input_name)):
            if pick_year == 0:
                temp_year = max_types_series[cluster_ind]['shap_{}'.format(var_name[var_ind])].iloc[0][input_name[input_ind]]
                NSC_year = NSC_list[cluster_ind][input_name[input_ind]+'_'+var_name[var_ind]]
                NSC_year.index = pd.to_datetime(temp_year.index)
                line = axes.plot(NSC_year.groupby(NSC_year.index.dayofyear).mean(), 
                      #label =input_name[input_ind]+unit[var_ind],
                          color=colors[len(colors)-input_ind-1], linewidth=1)
                
                axes.set_xlabel('DOY', fontsize=small_fontsize)#smallest_fotsz)
                axes.xaxis.set_major_locator(MultipleLocator(30*2)) 
                for tick in axes.xaxis.get_major_ticks():  
                    tick.label1.set_fontsize(small_fontsize)
                for tick in axes.yaxis.get_major_ticks():  
                    tick.label1.set_fontsize(small_fontsize)
                # MultipleLocator(30)
            else:
                temp_year = max_types_series[cluster_ind]['shap_{}'.format(var_name[var_ind])].iloc[0][input_name[input_ind]]
                temp_year.index = pd.to_datetime(temp_year.index)
                temp_year = temp_year[temp_year.index.year == pick_year].rolling(window = 5).mean().dropna().resample('5D').first()
                line = axes.plot(temp_year, 
                      #label =input_name[input_ind]+unit[var_ind],
                          color=colors[len(colors)-input_ind-1], linewidth=1)
                
                axes.set_xlabel('year of '+str(pick_year), fontsize=small_fontsize)#smallest_fotsz)
                
                monthFmt = mdates.DateFormatter('%b')
                axes.xaxis.set_major_locator(mdates.MonthLocator())
                axes.xaxis.set_major_formatter(monthFmt)
                
#                axes.set_xticks(temp_year[temp_year.index.year == pick_year].index)
#                axes.set_xticklabels(temp_year.groupby(temp_year.index.dayofyear).mean().index, rotation=45)
            line_list.append(line)
        
        if cluster_ind == 0:
            axes.set_ylabel('Normalized Sensitivity\nCoefficient [-] ('+sub_title[var_ind]+')',
                            fontsize=small_fontsize)#smallest_fotsz)
        #axes.set_ylim([0, 1])
        
        temp = max_types_series[cluster_ind]['shap_{}'.format(var_name[var_ind])].iloc[0]
        temp.index = pd.to_datetime(temp.index)
        #axes.xaxis.set_major_locator(xmajorLocator) 
        
        if var_ind == 0:
            axes.set_title('('+chr(97+i)+') {} - {}°N {}°E'.format(GPI[cluster_ind],lat, lon),
                           fontsize = small_fontsize)#smallest_fotsz)
        else:
            axes.set_title('('+chr(97+i)+')',
                           fontsize = small_fontsize)#smallest_fotsz)
        # add legend outside the subplots
        #lg_ax = fig.add_axes([0.15, 0.15, 0.7, 0.05])
        
        i = i+1
    #h,l = axes.get_legend_handles_labels()
    lgd = fig.legend(line_list[0:4],     # The line objects [line1, line2, line3, line4]
               labels=input_name,   # The labels for each line
               fontsize = small_fontsize,
               borderaxespad=0.1,    # Small spacing around legend box
               ncol=4,
               borderpad=1.5, labelspacing=1.5,
               loc="upper center",
               bbox_to_anchor=(0.5-0.07/2,0.1-0.01))
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.subplots_adjust(left=0.07, right=0.93, wspace=0.15, hspace=0.35,
                        bottom = 0.15)
    plt.savefig('{}{}'.format(save_at,fig_name+'.eps'),bbox_extra_artists=(lgd,), bbox_inches='tight',format = 'eps')
    plt.savefig('{}{}'.format(save_at,fig_name+'.jpeg'),bbox_extra_artists=(lgd,), bbox_inches='tight',format = 'jpeg')
    
    plt.show()
    #plt.close()
input_name = ['LAI','WG2','GPP','RE']
var_name = ['sig', 'slope', 'curv']
sub_title= [r'$\sigma_{40}^o$ [dB]',r"$\sigma^{\prime}$ [dB/deg]",r'$\sigma^{\prime\prime}{\rm[dB/deg^2]}$']

GPI = ['Broadleaf', 'Agriculture','Grassland', 'Needleleaf']
unit = ['[dB]', '[dB/deg]', '[dB/deg]'] # double check the unit for curv!
save_at = 'E:/Surfsara Output DNN/DNN Run Output Pickle/France/IEEEplots/'

xmajorLocator   = MultipleLocator(30)#MultipleLocator(2*365)
ymajorLocator   = MultipleLocator(2)
xminorLocator   = MultipleLocator(1)
yminorLocator   = MultipleLocator(1)

plotting_time_types(NSC_list, 0, input_name, var_name, 
                    "Fig12_inputSens_climatology", 
                        unit, GPI, save_at)
#%% plot of sensitivity to hyperparameters
# using iteration process of bayesian optimization process
#for i 