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
from matplotlib.colors import from_levels_and_colors
#import nmmn.plots
import shapefile
import pandas as pd
from PIL import Image


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
    
def plot_cart_obj(background, xp, yp, sp, fp, xn, yn, sn, fn, cp, cn, imtitle, savepath, imname, radar):
    """
    plots cartesian reflectivity and radar locations and detected mesocyclones

    Parameters
    ----------
    background : 2D array
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
    fig=plt.figure(figsize=(6.4,7.1),frameon=False)#figsize=(14,10)

    o_x=254000
    o_y=-159000
    xp = (xp - o_x)/1000
    xn = (xn - o_x)/1000
    yp = (yp - o_y)/1000
    yn = (yn - o_y)/1000
    
    bounds = list([4, 8] + np.linspace(13, 61, 17, dtype=int).tolist())  

    colors = [
        "#d3ebff",
        "#e9d7f3",
        "#9d7f95",
        "#650165",
        "#af01af",
        "#3333c9",
        "#0165ff",
        "#019797",
        "#02c933",
        "#65ff01",
        "#97ff01",
        "#c9ff33",
        "#ffff01",
        "#ffc901",
        "#ffa101",
        "#ff7d01",
        "#e11901",
        "#c10101",
        "#9f0101",
    ]


    cmap, _ = from_levels_and_colors(bounds,colors,extend='max')
    norm = mpl.colors.BoundaryNorm(bounds, ncolors=len(bounds) - 1)
    
    #cmap=plt.cm.turbo
    #cmap.set_under(color='gray')
    p0=plt.imshow(background, cmap=cmap, norm=norm)
    # ax = plt.axes([0,0,1,1])
    # plt.colorbar(p0, cmap=cmap, extend='both', orientation='vertical',shrink=0.7, boundaries=[0,1,2,5,10,15,25,40,60,100,150,250], ticks=[0,1,2,5,10,15,25,40,60,100,150,250])
    p1=plt.scatter((np.array(radar["x"])- o_x)/1000,(np.array(radar["y"])- o_y)/1000,s=5,c='black',marker=".")

    ap=np.ones(len(fp)); ap[fp==0]=0.8
    an=np.ones(len(fn)); an[fn==0]=0.8
    ccp=np.round((cp.values+1)*fp.values).astype(int); ccp[ccp>5]=5
    ccn=np.round((cn.values+1)*fn.values).astype(int); ccn[ccn>5]=5
    color=np.array(['grey','white','green','darkorange','firebrick','purple'])
    if len(xp)>0:
      p2=plt.scatter(xp,yp,s=30,c=color[ccp], vmin=0, vmax=5, marker="^",edgecolors='aqua')
    if len(xn)>0:
      p3=plt.scatter(xn,yn,s=30,c=color[ccn], vmin=0, vmax=5, marker="v",edgecolors='red')
    plt.axis('off')
    plt.ylim(0, 640)
    plt.xlim(0, 710)
    # plt.tight_layout()
    namefig=savepath + imname
    fig.patch.set_visible(False)
    # plt.show()
    # with open(namefig, 'wb') as outfile:
    #     fig.canvas.print_png(outfile)
    plt.savefig(namefig,transparent=True,bbox_inches='tight',dpi=300,pad_inches=0)
    plt.close()

def plot_cart_hist(time,background,trtcells,vert_p,vert_n, imtitle, savepath, imname, radar):
    """
    plots cartesian reflectivity and radar locations and detected mesocyclones and 2h history

    Parameters
    ----------

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
    o_x=254000
    o_y=-159000
    # background=np.zeros([640,710]); background[:]=np.nan
    fig=plt.figure(figsize=(7.1,6.4),frameon=False)#figsize=(14,10)
    # cmap=plt.cm.turbo
    
    bounds = [0.00, 0.01, 0.16, 0.25, 0.40,
              0.63, 1.00, 1.60, 2.50, 4.00,
              6.30, 10.00, 16.00, 25.00, 40.00,
              63.00, 100.]
    bounds = list([4, 8] + np.linspace(13, 61, 17, dtype=int).tolist())  

    colors = [
        "#d3ebff",
        "#e9d7f3",
        "#9d7f95",
        "#650165",
        "#af01af",
        "#3333c9",
        "#0165ff",
        "#019797",
        "#02c933",
        "#65ff01",
        "#97ff01",
        "#c9ff33",
        "#ffff01",
        "#ffc901",
        "#ffa101",
        "#ff7d01",
        "#e11901",
        "#c10101",
        "#9f0101",
    ]

    cmap, _ = from_levels_and_colors(bounds,colors,extend='max')
    norm = mpl.colors.BoundaryNorm(bounds, ncolors=len(bounds) - 1)
    
    p0=plt.imshow(background, cmap=cmap, norm=norm)
    p1=plt.scatter((np.array(radar["x"])- o_x)/1000,(np.array(radar["y"])- o_y)/1000,s=5,c='black',marker=".")
    idds=pd.concat([vert_p,vert_n])
    ids=np.unique(idds.ID).astype(int)
    for t_id in ids:
        tcell=trtcells[trtcells.traj_ID.astype(int)==t_id]

        # select rotations of the current cell without the current timestep
        pcell=vert_p[(vert_p.ID.astype(int)==t_id) & (vert_p.time.astype(int)!=int(time))]
        ncell=vert_n[(vert_n.ID.astype(int)==t_id) & (vert_n.time.astype(int)!=int(time))]

        # if np.nansum(pcell.flag)+np.nansum(ncell.flag)==0: continue
        xp = (pcell.x.astype(float) - o_x)/1000
        xn = (ncell.x.astype(float) - o_x)/1000
        xt = (tcell.chx.astype(float) - o_x)/1000
        yp = (pcell.y.astype(float) - o_y)/1000
        yn = (ncell.y.astype(float) - o_y)/1000
        yt = (tcell.chy.astype(float) - o_y)/1000
        print('rot')

        sp=np.nansum([pcell.A_n,pcell.D_n,pcell.L_n,pcell.P_n,pcell.W_n]);fp=pcell.flag;cp=pcell.rank_90
        sn=np.nansum([ncell.A_n,ncell.D_n,ncell.L_n,ncell.P_n,ncell.W_n]);fn=ncell.flag;cn=ncell.rank_90
        
        ap=np.ones(len(fp)); ap[fp==0]=0.8
        an=np.ones(len(fn)); an[fn==0]=0.8
        dp= np.nansum(pcell.dist)>0; dn= np.nansum(ncell.dist)>0
        fp= dp; fn=dn # int((len(xp)>3) * dp); fn= int((len(xn)>3) * dn) #only grey if too close to radar
        ccp=np.round((cp.values+1)*fp).astype(int); ccp[ccp>5]=5
        ccn=np.round((cn.values+1)*fn).astype(int); ccn[ccn>5]=5
        
        p4 = plt.plot(xt,yt,color='black',linewidth=0.5)
        
        color=np.array(['grey','aliceblue','green','darkorange','firebrick','purple'])
        if len(xp)>0:
          p2=plt.scatter(xp,yp,s=20,c=color[ccp], vmin=0, vmax=5, marker=r'$\circlearrowleft$',edgecolors='aqua',linewidth=0.15)#,alpha=0.8)
        if len(xn)>0:
          p3=plt.scatter(xn,yn,s=20,c=color[ccn], vmin=0, vmax=5, marker=r'$\circlearrowright$',edgecolors='grey',linewidth=0.15)#,alpha=0.5)
    
    # Selects rotation of current timestep and plots it larger      
    pcell=vert_p[vert_p.time.astype(int)==int(time)]
    ncell=vert_n[vert_n.time.astype(int)==int(time)]
    
    xp = (pcell.x.astype(float) - o_x)/1000
    xn = (ncell.x.astype(float) - o_x)/1000
    yp = (pcell.y.astype(float) - o_y)/1000
    yn = (ncell.y.astype(float) - o_y)/1000
    print('rot')
 
    sp=np.nansum([pcell.A_n,pcell.D_n,pcell.L_n,pcell.P_n,pcell.W_n]);fp=pcell.flag;cp=pcell.rank_90
    sn=np.nansum([ncell.A_n,ncell.D_n,ncell.L_n,ncell.P_n,ncell.W_n]);fn=ncell.flag;cn=ncell.rank_90
    
    ap=np.ones(len(fp)); ap[fp==0]=0.8
    an=np.ones(len(fn)); an[fn==0]=0.8
    dp= pcell.dist>0; dn= ncell.dist>0
    fp=dp; fn=dn #fp= int((len(xp)>3) * dp); fn= int((len(xn)>3) * dn)
    ccp=np.round((cp.values+1)*fp).astype(int); ccp[ccp>5]=5
    ccn=np.round((cn.values+1)*fn).astype(int); ccn[ccn>5]=5
    
    
    color=np.array(['grey','white','green','darkorange','firebrick','purple'])
    if len(xp)>0:
      p2=plt.scatter(xp,yp,s=45,c=color[ccp], vmin=0, vmax=5, marker=r'$\circlearrowleft$',edgecolors='aqua',linewidth=0.3)#,alpha=0.8)
    if len(xn)>0:
      p3=plt.scatter(xn,yn,s=45,c=color[ccn], vmin=0, vmax=5, marker=r'$\circlearrowright$',edgecolors='grey',linewidth=0.3)#,alpha=0.5)
    plt.axis('off')
    plt.ylim(0, 640)
    plt.xlim(0, 710)
    # plt.tight_layout()
    namefig=savepath + imname
    fig.patch.set_visible(False)
    # plt.show()
    # with open(namefig, 'wb') as outfile:
    #     fig.canvas.print_png(outfile)

    plt.savefig(namefig,transparent=True,bbox_inches='tight',dpi=259.8,pad_inches=0)
    plt.close()

    # Resize the image to be sure
    img = Image.open(namefig)
    resized_img = img.resize((1420, 1280), Image.LANCZOS)

    # Overwrite the original file with the resized image
    resized_img.save(namefig)
    
def plot_cart_day(trtcells,vert_p,vert_n, imtitle, savepath, imname, radar):
    """
    plots daily TRT tracks and rotation locations

    Parameters
    ----------

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
    o_x=254000
    o_y=-159000
    background=np.zeros([640,710]); background[:]=np.nan
    fig=plt.figure(figsize=(6.4,7.1),frameon=False)#figsize=(14,10)
    p0=plt.imshow(background, origin='lower')
    p1=plt.scatter((np.array(radar["x"])- o_x)/1000,(np.array(radar["y"])- o_y)/1000,s=5,c='black',marker=".")
    print(len(trtcells))
    if len(trtcells)>0:
      ids=np.unique(trtcells.traj_ID).astype(int)
      for t_id in ids:
          tcell=trtcells[trtcells.traj_ID.astype(int)==t_id]
          pcell=vert_p[vert_p.ID.astype(int)==t_id]
          ncell=vert_n[vert_n.ID.astype(int)==t_id]
          
          sp=np.nansum([pcell.A_n,pcell.D_n,pcell.L_n,pcell.P_n,pcell.W_n]);fp=pcell.flag;cp=pcell.rank_90
          sn=np.nansum([ncell.A_n,ncell.D_n,ncell.L_n,ncell.P_n,ncell.W_n]);fn=ncell.flag;cn=ncell.rank_90
          
          dp= np.nansum(pcell.dist)>0; dn= np.nansum(ncell.dist)>0
          fp= int((len(pcell)>3) * dp); fn= int((len(ncell)>3) * dn)
          ccp=np.round((cp.values+1)*fp).astype(int); ccp[ccp>5]=5
          ccn=np.round((cn.values+1)*fn).astype(int); ccn[ccn>5]=5
          
          if fp+fn==0: continue
          xp = (pcell.x.astype(float) - o_x)/1000
          xn = (ncell.x.astype(float) - o_x)/1000
          xt = (tcell.chx.astype(float) - o_x)/1000
          yp = (pcell.y.astype(float) - o_y)/1000
          yn = (ncell.y.astype(float) - o_y)/1000
          yt = (tcell.chy.astype(float) - o_y)/1000
          print('rot')
  
  
          
          #p4 = plt.plot(xt,yt,color='black',linewidth=0.5)
          
          #ap=np.ones(len(fp)); ap[fp==0]=0.8
          #an=np.ones(len(fn)); an[fn==0]=0.8
          
          
          ccp=np.round((cp.values+1)).astype(int); ccp[ccp>5]=5
          ccn=np.round((cn.values+1)).astype(int); ccn[ccn>5]=5
          color=np.array(['grey','aliceblue','green','darkorange','firebrick','purple'])
          if len(xp)>2:
            p2=plt.scatter(xp,yp,s=20,c=color[ccp], vmin=0, vmax=5, marker=r'$\circlearrowleft$',edgecolors='aqua',linewidth=0.15)#,alpha=0.8)
            p4 = plt.plot(xt,yt,color='black',linewidth=0.5)
          if len(xn)>2:
            p3=plt.scatter(xn,yn,s=20,c=color[ccn], vmin=0, vmax=5, marker=r'$\circlearrowright$',edgecolors='grey',linewidth=0.15)#,alpha=0.5)
            p4 = plt.plot(xt,yt,color='black',linewidth=0.5)
    plt.axis('off')
    plt.ylim(0, 640)
    plt.xlim(0, 710)
    # plt.tight_layout()
    namefig=savepath + imname
    fig.patch.set_visible(False)
    # plt.show()
    # with open(namefig, 'wb') as outfile:
    #     fig.canvas.print_png(outfile)
    plt.savefig(namefig,transparent=True,bbox_inches='tight',dpi=259.8,pad_inches=0)
    print('saving figure',namefig)
    plt.close()

    # Resize the image to be sure
    img = Image.open(namefig)
    resized_img = img.resize((1420, 1280), Image.LANCZOS)

    # Overwrite the original file with the resized image
    resized_img.save(namefig)

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
    cmap=plt.cm.turbo
    cmap.set_under(color='gray')
    p0=plt.pcolormesh(myfinaldata, vmin=0, vmax=60, cmap=cmap)
    plt.colorbar(p0, cmap=cmap,  boundaries=np.arange(0,70,5), ticks=np.arange(0,70,5), extend='both', orientation='vertical',shrink=0.7)
    p1=plt.scatter((np.array(radar["x"])- o_x)/1000,(np.array(radar["y"])- o_y)/1000,s=None,c='black')
    borders = shapefile.Reader('/scratch/lom/mof/code/ELDES_MESO/map_radar/Border_CH.shp')
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
    fig.savefig(namefig,transparent=True)
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
