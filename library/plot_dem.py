#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:36:02 2020

@author: mfeldman
"""

import numpy as np
import matplotlib as mpl
mpl.use('module://ipykernel.pylab.backend_inline')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LightSource 
import matplotlib.patheffects as path_effects
import sys
import nmmn.plots
sys.path.append('/users/mfeldman/scripts/mesocyclone_detection')
sys.path.append('/home/feldmann/Documents/Research/CSCS_downloads/py_scripts/rotation')
from osgeo import gdal
import shapefile

#%%

def plot_density_dem(density, vmin, vmax, step, path, figname):
    #%% IMPORT DEM
    DOMAIN =  'CH'
    ##############################################################################
    
    
    # CONSTANTS
    YLIM = {}
    XLIM = {}
    YLIM['CH'] = [50000,320000]
    XLIM['CH']= [450000,850000]
    XLIM['J'] = [520200,608000]
    YLIM['J']  = [180000,250000]
    XLIM['T']  = [520200,608000]
    YLIM['T']  = [180000,250000]
    XLIM['E']  = [592000,670000]
    YLIM['E']  = [174000,226000]
    XLIM['M']  = [631000,637000]
    YLIM['M']  = [186000,192700]
    XLIM['H']  = [643000,651000]
    YLIM['H']  = [199000,206000]
    ##############################################################################
    
    
    
    if type(DOMAIN) == str:
        xlim = XLIM[DOMAIN]
        ylim = YLIM[DOMAIN]
    
    elif type(DOMAIN) in [list, tuple]:
        xlim = DOMAIN[0]
        ylim = DOMAIN[1]
        
    offset = 1/50. * (ylim[1]-ylim[0])
    dempath='/users/mfeldman/map_radar/'
    
    dem = gdal.Open(dempath+'/dem/dem_100m_ch.tif')
    width = dem.RasterXSize
    height = dem.RasterYSize
    gt = dem.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width*gt[4] + height*gt[5] 
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3]
    
    allx =  np.linspace(minx,maxx,width)
    ally = np.linspace(miny,maxy,height)
    x,y = np.meshgrid(allx,ally)
    all_z = dem.ReadAsArray()
    all_z[all_z<0] = np.nan
    
    ls = LightSource(azdeg=315, altdeg=45)
    cmap = plt.cm.terrain
    
    dx =1 
    dy=1
    
    fig=plt.figure(figsize=(16.8,10.))
    ax = plt.gca()
    cmap1 = plt.cm.gist_gray_r

    cmap1.set_bad('black')
    cmap1.set_gamma(1.)
    
    hs = ls.hillshade(all_z, vert_exag=0.02, dx=dx, dy=dy)
    
    rgb = ls.shade(all_z, cmap=cmap1, blend_mode='soft',
                           vert_exag=0.02, dx=dx, dy=dy, vmin=-1000, vmax=4000)
    
    lakes = hs==hs[0,0]
    lakes[:,1:]+=lakes[:,0:-1]
    lakes[:,1:]+=lakes[:,0:-1]
    lakes[:,0:-1]+=lakes[:,1:]
    
    rgb[lakes] = 	[65/255., 105/255., 225/255.,1]
    
    mu = ax.imshow(rgb, vmin=-1000, vmax=4000, extent=[minx,maxx,miny,maxy])
    im = ax.imshow(all_z, cmap=cmap1, vmin=-1000, vmax=4000)
    
    v = np.arange(0,4600,400)
    b = np.arange(0,4010,10)
    a = fig.colorbar(im,fraction=0.029, pad=0.06, label='Altitude ASL [m]',ticks=v, boundaries=b)
    a.ax.tick_params(labelsize=13)
    a.set_label(label='Altitude ASL [m]',size=13)
    im.remove()
    o_x=255000
    o_y=-160000
    data_x=np.arange(o_x,o_x+710000,1000)
    data_y=np.arange(o_y,o_y+640000,1000)
    data_xx,data_yy=np.meshgrid(data_x,data_y)
    turbo=nmmn.plots.turbocmap()
    cmap=turbo
    p0=plt.imshow(np.flipud(density), vmin=vmin, vmax=vmax, cmap=cmap, alpha=.66,extent=[o_x,o_x+710000,o_y,o_y+640000])
    #p0=plt.pcolormesh(data_xx,data_yy,density, vmin=vmin, vmax=vmax, cmap=cmap, alpha=0.2)
    p=fig.colorbar(p0,fraction=0.029, pad=0.04,label='storms per 100 km$^2$', cmap=cmap, boundaries=np.arange(vmin-step,vmax+step+1,step), ticks=np.arange(vmin,vmax+1,step), extend='both')
    p.ax.tick_params(labelsize=13)
    p.set_label(label='storms per 100 km$^2$',size=13)
    borders = shapefile.Reader(dempath+'Border_CH.shp')
    listx=[]
    listy=[]
    for shape in borders.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        listx.append(x);listy.append(y)
        ax.plot(x,y,'r',linewidth=1.5)
    name_file = dempath+'data_radars.csv'
    pos = np.genfromtxt(name_file, delimiter = ';', usecols=[2,3], dtype=float,skip_header = 1)
    ax.scatter(pos[:,0],pos[:,1], s = 100, c = 'red')
    ax.tick_params(axis='both', labelsize=13)
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_xlabel('Swiss X-coordinate [m]', fontsize = 16)
    ax.set_ylabel('Swiss Y-coordinate [m]', fontsize = 16)
    xlim=[o_x,o_x+711000]
    ylim=[o_y,o_y+640000]
    xlim=[minx+15000,maxx-50000]
    ylim=[miny+8000,maxy-100000]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    fig.savefig(path+figname, dpi = 80, bbox_inches = 'tight')
    
    #%%
def plot_density(density, dem, imtitle, savepath, imname, radar, vmin, vmax):
    ## plots a cartesian plot of a grid of x,y data
    # x: meshgrid of x
    # y: meshgrid of y
    # myfinaldata: gridded data
    # vmin: minimal value, vmax: maximal value, cmap: colormap, bound: bounds for colormap
    # imtitle: title, savepath: path of image, imname: name of saved image
    o_x=255000
    o_y=-160000
    fig=plt.figure(figsize=(14.2,12.8))
#    ax=plt.subplot(1, 1, 1)
    turbo=nmmn.plots.turbocmap()
    # cmap=cc.cm.CET_L16
    # cmap.set_bad(color='gray')
    p=plt.pcolormesh(dem, vmin=0, vmax=4000, cmap='Greys_r')
    cmap=turbo
    # cmap.set_bad(color='gray')
    p0=plt.pcolormesh(density, vmin=vmin, vmax=vmax, cmap=cmap, alpha=0.2)
    plt.colorbar(p0, cmap=cmap,  boundaries=np.arange(vmin,vmax+1,5), ticks=np.arange(vmin,vmax+1,5), extend='both', orientation='horizontal', shrink=0.5)
    p1=plt.scatter((np.array(radar["x"])- o_x)/1000,(np.array(radar["y"])- o_y)/1000,s=None,c='black')
    plt.title(imtitle)
    plt.ylim(0, 640)
    plt.xlim(0, 710)
    namefig=savepath + imname
    plt.show()
    fig.savefig(namefig)
    plt.close(fig=fig)
def plot_diff(diff, imtitle, savepath, imname, radar, vmin, vmax):
    ## plots a cartesian plot of a grid of x,y data
    # x: meshgrid of x
    # y: meshgrid of y
    # myfinaldata: gridded data
    # vmin: minimal value, vmax: maximal value, cmap: colormap, bound: bounds for colormap
    # imtitle: title, savepath: path of image, imname: name of saved image
    o_x=255000
    o_y=-160000
    fig=plt.figure(figsize=(14.2,12.8))
    cmap=plt.cm.RdBu_r;
    cmap.set_bad(color='gray')
    p0=plt.pcolormesh(diff, vmin=vmin, vmax=vmax, cmap=cmap, norm=colors.LogNorm(vmin, vmax))
    # p0=plt.pcolormesh(density, vmin=vmin, vmax=vmax, cmap=cmap, norm=colors.LogNorm(vmin, vmax))
    plt.colorbar(p0, extend='both', orientation='horizontal', shrink=0.5)
    p1=plt.scatter((np.array(radar["x"])- o_x)/1000,(np.array(radar["y"])- o_y)/1000,s=None,c='black')
    plt.title(imtitle)
    plt.ylim(0, 640)
    plt.xlim(0, 710)
    namefig=savepath + imname
    plt.show()
    fig.savefig(namefig)
    plt.close(fig=fig)