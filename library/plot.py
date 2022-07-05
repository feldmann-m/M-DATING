#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:05:39 2020

@author: feldmann
"""


import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#import nmmn.plots
import shapefile


def plot_ppi_MF_masked(theta, r, myfinaldata, vmin, vmax, cmap, bound, imtitle, savepath, imname):
    """
    plots polar data

    Parameters
    ----------
    theta : 2D array
        azimuth angles of all datapoints.
    r : 2D array
        radius of all datapoints.
    myfinaldata : 2D array
        data to be plotted.
    vmin : float
        minimum value.
    vmax : float
        maximum value.
    cmap : string
        matplotlib colormap.
    bound : array
        bounds for colormap.
    imtitle : string
        image caption.
    savepath : string
        path.
    imname : string
        file name.

    Returns
    -------
    None.

    """
    fig=plt.figure(figsize=(12,12))
    ax=plt.subplot(1, 1, 1, projection='polar')
    ax.set_theta_offset(0.5*np.pi)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(315)
    cmap=cmap
    cmap.set_bad(color='gray')
    p1=plt.pcolormesh(theta, r, myfinaldata, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(p1, cmap=cmap,  boundaries=bound, ticks=bound, extend='both')
    plt.grid()
    plt.title(imtitle)
    namefig=savepath + imname
    plt.show()
    fig.savefig(namefig)
    plt.close(fig=fig)
    
def plot_cart_scatter(myfinaldata, xp, yp, sp, xn, yn, sn, colorp, colorn, contours, imtitle, savepath, imname, radar):
    """
    plots cartesian reflectivity and radar locations and detected mesocyclones

    Parameters
    ----------
    myfinaldata : 2D array
        cartesian reflectivity data.
    xp : array
        x coordinates of positive rotation.
    yp : array
        y coordinates of positive rotation.
    sp : array
        size of positive rotation.
    xn : array
        x coordinates of negative rotation.
    yn : array
        y coordinates of negative rotation.
    sn : array
        size of negative rotation.
    colorp : array
        strength of positive rotation.
    colorn : array
        strength of negative rotation.
    contours : list
        list of thunderstorm contours.
    imtitle : string
        image title.
    savepath : string
        path.
    imname : string
        filename.
    radar : dict
        radar meta data.

    Returns
    -------
    None.

    """
    fig=plt.figure(figsize=(14,10))

    o_x=254000
    o_y=-159000
    xp = (xp - o_x)/1000
    xn = (xn - o_x)/1000
    yp = (yp - o_y)/1000
    yn = (yn - o_y)/1000
    #turbo=nmmn.plots.turbocmap()
    cmap='turbo'
    cmap.set_under(color='gray')
    p0=plt.pcolormesh(myfinaldata, vmin=0, vmax=60, cmap=cmap)
    plt.colorbar(p0, cmap=cmap,  boundaries=np.arange(0,70,5), ticks=np.arange(0,70,5), extend='both', orientation='vertical',shrink=0.7)
    p1=plt.scatter((np.array(radar["x"])- o_x)/1000,(np.array(radar["y"])- o_y)/1000,s=None,c='black')
    borders = shapefile.Reader('/users/mfeldman/map_radar/Border_CH.shp')
    listx=[]
    listy=[]
    for shape in borders.shapeRecords():
        x = [(i[0]-o_x)/1000 for i in shape.shape.points[:]]
        y = [(i[1]-o_y)/1000 for i in shape.shape.points[:]]
        listx.append(x);listy.append(y)
        plt.plot(x,y,'r',linewidth=1.5)
    p2=plt.scatter(xp,yp,s=sp,c=colorp, vmin=8, vmax=27, cmap='Blues', edgecolors='gray')
    p3=plt.scatter(xn,yn,s=sn,c=colorn, vmin=8, vmax=27, cmap='Reds', edgecolors='gray')
    for contour in contours:
        p4=plt.plot(contour[:,1], contour[:,0], color='grey')
    # plt.colorbar(p2, cmap='Blues',  boundaries=np.arange(9, 27), ticks=np.arange(9, 27), extend='both')
    # plt.colorbar(p3, cmap='Reds',  boundaries=np.arange(9, 27), ticks=np.arange(9, 27), extend='both')
    plt.title(imtitle)
    plt.ylim(0, 640)
    plt.xlim(0, 710)
    plt.tight_layout()
    namefig=savepath + imname
    plt.show()
    fig.savefig(namefig)
    plt.close(fig=fig)
    

def turbo_cm():
    """
    gets turbo colormap

    Returns
    -------
    Turbo_r : colormap
        turbo colormap.

    """
    from bokeh.palettes import Turbo
    Turbo_r=Turbo[256][::-1]
    return Turbo_r

def plot_track(track_list, imtitle, savepath, imname, radar):
    """
    plots thunderstorm tracks

    Parameters
    ----------
    track_list : list
        list of thunderstorm tracks.
    imtitle : string
        image title.
    savepath : string
        image path.
    imname : string
        image file name.
    radar : dict
        radar metadata.

    Returns
    -------
    None.

    """
    o_x=255000
    o_y=-160000
    fig=plt.figure(figsize=(14.2,12.8))
    plt.ylim(0, 640)
    plt.xlim(0, 710)
    p1=plt.scatter((np.array(radar["x"])- o_x)/1000,(np.array(radar["y"])- o_y)/1000,s=None,c='black')
    color=iter(plt.cm.spring(np.linspace(0,1,len(track_list))))
    for track in track_list:
        p2=plt.plot(track.max_x, track.max_y, c=next(color))
    plt.title(imtitle)
    namefig=savepath + imname
    fig.savefig(namefig)
    plt.close(fig=fig)

def plot_rottrack(track_list_p, track_list_n, imtitle, savepath, imname, radar):
    """
    plots rotation track

    Parameters
    ----------
    track_list_p : list
        list of positive rotation tracks.
    track_list_n : list
        list of negative rotation tracks.
    imtitle : string
        image title.
    savepath : string
        image path.
    imname : string
        image file name.
    radar : dict
        radar metadata.

    Returns
    -------
    None.

    """
    o_x=255000
    o_y=-160000
    fig=plt.figure(figsize=(14.2,12.8))
    
    p1=plt.scatter(np.array(radar["x"]),np.array(radar["y"]),s=None,c='black')
    borders = shapefile.Reader('/users/mfeldman/map_radar/Border_CH.shp')
    listx=[]
    listy=[]
    for shape in borders.shapeRecords():
        x = [(i[0]) for i in shape.shape.points[:]]
        y = [(i[1]) for i in shape.shape.points[:]]
        listx.append(x);listy.append(y)
        p4=plt.plot(x,y,'r',linewidth=1.5)
    n=len(track_list_p)
    color=iter(plt.cm.Blues_r(np.linspace(0,1,n+1)))
    for track in track_list_p:
        p2=plt.plot(track.x, track.y, c=next(color), linewidth=5)
    n=len(track_list_n)
    color=iter(plt.cm.Reds_r(np.linspace(0,1,n+1)))
    for track in track_list_n:
        p2=plt.plot(track.x, track.y, c=next(color), linewidth=5)
        # plt.legend(p2, track.ID[0])
    plt.title(imtitle)
    plt.ylim(o_y, o_y+640000)
    plt.xlim(o_x, o_x+710000)
    namefig=savepath + imname
    plt.show()
    fig.savefig(namefig)
    plt.close(fig=fig)
