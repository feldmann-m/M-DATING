#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:06:13 2020

@author: feldmann
"""

import pyart
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import copy
from datetime import datetime
#%%
def mask(mask, coord, radar, cartesian, r, el):
    """
    Generates polar binary mask for thunderstorm cells

    Parameters
    ----------
    mask : array
        Cartesian mask.
    coord : array
        origin coordinates.
    #radar : dict
        radar metadata.
    cartesian : dict
        cartesian metadata including radar coordinates.
    r : int
        radar number.
    el : int
        elevation number.

    Returns
    -------
    p_mask : array
        binary array with thunderstorm mask.

    """
    c_el=coord[el]
    cr_el=np.round(copy.deepcopy(c_el/1000)).astype(int)
    cr_el[0,:,:]+=cartesian["rx"][r]
    cr_el[0,:,:][cr_el[0,:,:]>=710]=709
    cr_el[1,:,:]+=cartesian["ry"][r]
    cr_el[1,:,:][cr_el[1,:,:]>=640]=639
    
    p_mask=mask[cr_el[1,:,:],cr_el[0,:,:]]
    p_mask=(p_mask*1).astype(float)
    p_mask[p_mask==0]=np.nan
    p_mask[:,:10]=np.nan
    
    return p_mask

def pattern_vectors(data, shear, distance):
    """
    Identification of 1D pattern vectors
    contiguous areas in azimuth exceeding shear threshold

    Parameters
    ----------
    data : array
        azimuthal shear.
    shear: dict
        contains shear thresholds.
    distance : array
        Cartesian azimuthal distance between bins.

    Returns
    -------
    shearID : array
        array with IDs in fields, where pattern vector was identified.

    """
    
    print("Identifying pattern vectors")
    shear_ID=np.zeros(data.shape)
    shearID=np.zeros(data.shape)
    IDpos=1
    for m in range(0,data.shape[1]):
        #print (m)
        for n in range(0,data.shape[0]-2):
            if n<=40: shear_thresh=shear["near1"]
            if n<200 and n>40: shear_thresh=shear["near1"]-(shear["near1"]-shear["far1"])*((n-40)/160)
            if n>=200: shear_thresh=shear["far1"]
            if all(data[n:n+3,m]>shear_thresh): shear_ID[n:n+3,m]=IDpos
            elif all(data[n-1:n+2,m]>shear_thresh) and data[n+2,m]<=shear_thresh: IDpos+=1
            
    for n in range(0,IDpos+1):
        indices=np.where(shear_ID==n)
        if np.sum(distance[indices])<shear["length"]:
            shear_ID[shear_ID==n]=0
            
    shearID[shear_ID>0]=1
    
    return shearID

def shear_group(rotation, sign, myfinaldata, az_shear, labels, resolution, distance, shear, radar, EL, el, R, r, fg_indices, time):
    """
    Merging pattern vectors to areas
    Computing intensity metrics of area

    Parameters
    ----------
    rotation : dict
        storage of rotation signatures.
    sign : int
        sign of rotation.
    myfinaldata : array
        velocity data.
    az_shear : array
        azimuthal velocity shear.
    labels : float
        thunderstorm ID.
    resolution : float
        radial resolution in km.
    distance : array
        azimuthal distance.
    shear : dict
        contains shear thresholds for object identification.
    radar : dict
        radar metadata.
    #EL : string
        elevation name.
    el : int
        elevation number.
    #R : string
        radar name.
    r : int
        radar number.
    #fg_indices : list## or rewrite into lookup table
        indices for cartesian transformation.
    time : int
        date-time stamp.

    Returns
    -------
    rotation : dict
        storage of all rotation signatures.

    """
    ##merging pattern vectors to areas
    min_width=shear["width"]
    az_shear=sign*az_shear
    myfinaldata=sign*myfinaldata
    shearID=pattern_vectors(az_shear, shear["far1"], 
                                            shear["near1"], 
                                            shear["length"], 
                                            distance)
    rotation["pattern_vector"]=shearID
    print("Identifying shear areas")
    shear_groups, g_ID = ndi.label(shearID)
    shear_prop=[]
    for n in range(1,g_ID+1):
        indices=np.where(shear_groups==n)
        if len(np.unique(indices[1]))>2:
            vertical_ID=labels
            size=np.nansum(distance[indices]*resolution)
            vol=np.nansum(distance[indices]*resolution*resolution)
            cen_r=np.mean(indices[1])
            if cen_r<=40:
                min_shear_2=shear["near2"]-shear["near2d"]*((40-cen_r)/40)
                min_rvel=shear["rvel1"]-shear["rvel1d"]*((40-cen_r)/40)
                min_vort=shear["vort1"]-shear["vort1d"]*((40-cen_r)/40)
            if cen_r<200 and cen_r>40:
                min_shear_2=shear["near2"]-shear["near2d"]*((cen_r-40)/160)
                min_rvel=shear["rvel1"]-shear["rvel1d"]*((cen_r-40)/160)
                min_vort=shear["vort1"]-shear["vort1d"]*((cen_r-40)/160)
            if cen_r>=200: min_shear_2=shear["near2"]-shear["near2d"]; min_rvel=shear["rvel1"]-shear["rvel1d"]; min_vort=shear["vort1"]-shear["vort1d"]
            maxshear=max(az_shear[indices],key=abs)
            binary=np.zeros(myfinaldata.shape)
            binary[:]=np.nan
            binary[indices]=1
            filt=binary*myfinaldata
            ma=np.nanmax(filt); mi=np.nanmin(filt)
            dvel=abs(ma-mi)/2
            lmax=np.where(filt==ma); lmin=np.where(filt==mi)
            xmax,ymax,zmax=pyart.core.antenna_to_cartesian(lmax[1], lmax[0], el)
            xmin,ymin,zmin=pyart.core.antenna_to_cartesian(lmin[1], lmin[0], el)
            dis=np.array([])
            for n in range(len(xmax)):
                dx=xmin-xmax[n]; dy=ymin-ymax[n]
                dis=np.append(dis,np.sqrt(dx*dx + dy*dy),axis=0)
            mindis=np.nanmin(dis)
            vort=4*dvel/mindis
            if np.isnan(dvel): dvel=0
            vec_width=[]
            for m in np.unique(indices[1]):
                a=len(np.where(indices[1]==m)[0])
                vec_width.append(distance[0,m]*a)
            maxwidth=np.nanmax(vec_width)
            maxlen=len(np.unique(indices[1]))
            ratio=maxlen/maxwidth
            rankvel=(dvel-min_rvel)/(min_rvel)
            rankvort=(vort-min_vort)/(4*min_vort)
            rank=np.nanmean([rankvel,rankvort])*5
            if maxlen<min_width or ratio<(1/3) or ratio>3 or vort<min_vort or dvel<min_rvel or mindis<1000 or mindis>10000:#or abs(maxshear)<min_shear_2  or av_ref<20
                shear_groups[shear_groups==n]=0
            else:
                print("rotation characteristics:", dvel, vort, dis)
                x=[]; y=[]; z=[];
                for n in range(len(indices[0])):
                    cart_indices=pyart.core.antenna_to_cartesian(indices[1][n]/2, indices[0][n], el)
                    x.append(cart_indices[0])
                    y.append(cart_indices[1])
                    z.append(cart_indices[2])
                cart_centroids = np.mean(x), np.mean(y), np.mean(z)
                shear_prop.append([n, time, el, R, indices, cart_centroids[0]+radar["x"][r], cart_centroids[1]+radar["y"][r], cart_centroids[2]+radar["z"][r], dvel, vort, mindis, rank, vertical_ID, size, vol, cen_r])
                    
    sheargroups=np.zeros(myfinaldata.shape); sheargroups[:]=np.nan
    sheargroups[shear_groups>0]=1
    shear_prop=pd.DataFrame(data=shear_prop, columns=["ID", "time", "elevation", "radar", "indices", "x", "y", "z", "dvel", "vort", "diam", "rank", "v_ID", "size", "vol", "range"])
    print(shear_prop)
    print("Identified shear areas: ", len(shear_prop))
    rotation["prop"]=pd.concat([rotation["prop"],shear_prop], ignore_index=True)
    
    
    return rotation

def tower(rotation, areas, radar, shear, r, time, path):
    """
    Merging of identified objects within the same thunderstorm cell

    Parameters
    ----------
    rotation : dict
        all identified rotation objects.
    areas : array
        thunderstorm IDs.
    #radar : dict
        radar metadata.
    shear : dict
        contains vertical continuity thresholds.
    #r : int
        radar number.
    time : int
        date-time-stamp.
    path : dict
        path definitions.

    Returns
    -------
    towers : pandas dataframe
        rotation metrics for analyzed timestep sorted by thunderstorm ID.
    v_ID : array
        thunderstorm IDs with rotation.

    """
    ##merging vertically close objects
    print("vertical towers")
    if len(rotation["prop"])<2:
        towers=pd.DataFrame(columns=["ID", "time", "radar","x", "y", "dz",
                                "A","D","L","P","W","A_range","D_range","L_range","P_range","W_range","A_n","D_n","L_n","P_n","W_n","A_el","D_el","L_el","P_el","W_el",
                                "size_sum","size_mean","vol_sum","vol_mean",
                                "z_0","z_10", "z_25","z_50","z_75","z_90","z_100","z_IQR","z_mean",
                                "r_0","r_10", "r_25","r_50","r_75","r_90","r_100","r_IQR","r_mean",
                                "v_0","v_10", "v_25","v_50","v_75","v_90","v_100","v_IQR","v_mean",
                                "d_0","d_10", "d_25","d_50","d_75","d_90","d_100","d_IQR","d_mean",
                                "rank_0","rank_10", "rank_25","rank_50","rank_75","rank_90","rank_100","rank_IQR","rank_mean",
                                ])
        v_ID=0
        return towers, v_ID
    
    prop=rotation["prop"].copy()
    v_ID=prop["v_ID"].values
    print("building towers")
    n=np.unique(prop["v_ID"].values)
    print(n)
    towers=pd.DataFrame(data=0.0, index=n, columns=["ID", "time", "radar","x", "y", "dz",
                                "A","D","L","P","W","A_range","D_range","L_range","P_range","W_range","A_n","D_n","L_n","P_n","W_n","A_el","D_el","L_el","P_el","W_el",
                                "size_sum","size_mean","vol_sum","vol_mean",
                                "z_0","z_10", "z_25","z_50","z_75","z_90","z_100","z_IQR","z_mean",
                                "r_0","r_10", "r_25","r_50","r_75","r_90","r_100","r_IQR","r_mean",
                                "v_0","v_10", "v_25","v_50","v_75","v_90","v_100","v_IQR","v_mean",
                                "d_0","d_10", "d_25","d_50","d_75","d_90","d_100","d_IQR","d_mean",
                                "rank_0","rank_10", "rank_25","rank_50","rank_75","rank_90","rank_100","rank_IQR","rank_mean",
                                ])
    for ID in n:
        obj=prop.where(prop["v_ID"]==ID).dropna()
        if len(obj)<1: continue
        towers["ID"][ID]=ID
        towers["radar"][ID]=np.unique(obj["radar"].values)
        r_range=[]
        r_elev=[]
        r_n=[]
        for n in np.unique(obj["radar"].values):
            o=obj.loc[obj["radar"]==n]
            r_range.append(np.average(o["range"], weights=o["size"])*0.5)
            r_n.append(len(o["vol"]))
            r_elev.append(len(np.unique(o["elevation"])))
            if n == 'A':
                towers["A"][ID]=1
                towers["A_range"][ID]=np.average(o["range"], weights=o["size"])*0.5
                towers["A_n"][ID]=len(o["vol"])
                towers["A_el"][ID]=len(np.unique(o["elevation"]))
            if n == 'D':
                towers["D"][ID]=1
                towers["D_range"][ID]=np.average(o["range"], weights=o["size"])*0.5
                towers["D_n"][ID]=len(o["vol"])
                towers["D_el"][ID]=len(np.unique(o["elevation"]))
            if n == 'L':
                towers["L"][ID]=1
                towers["L_range"][ID]=np.average(o["range"], weights=o["size"])*0.5
                towers["L_n"][ID]=len(o["vol"])
                towers["L_el"][ID]=len(np.unique(o["elevation"]))
            if n == 'P':
                towers["P"][ID]=1
                towers["P_range"][ID]=np.average(o["range"], weights=o["size"])*0.5
                towers["P_n"][ID]=len(o["vol"])
                towers["P_el"][ID]=len(np.unique(o["elevation"]))
            if n == 'W':
                towers["W"][ID]=1
                towers["W_range"][ID]=np.average(o["range"], weights=o["size"])*0.5
                towers["W_n"][ID]=len(o["vol"])
                towers["W_el"][ID]=len(np.unique(o["elevation"]))
                
        ra=np.nanmin([towers.A_range,towers.D_range,towers.L_range,towers.P_range,towers.W_range])
        if ra>100:
            dz_min=shear["zu"]-shear["zd"]
            vort_max=shear["vortu"]-shear["vortd"]
            rot_max=shear["rotu"]-shear["rotd"]
        if ra>20 and ra<=100:
            dz_min=shear["zu"]-shear["zd"]*((ra-20)/80)
            vort_max=shear["vortu"]-shear["vortd"]*((ra-20)/80)
            rot_max=shear["rotu"]-shear["rotd"]*((ra-20)/80)
        if ra<=20:
            dz_min=shear["zu"]-shear["zu"]*((20-ra)/20)
            vort_max=shear["vortu"]-shear["vortd"]*((20-ra)/20)
            rot_max=shear["rotu"]-shear["rotd"]*((20-ra)/20)
        print(ra,dz_min,vort_max,rot_max)
        towers["dz"][ID]=max(obj["z"])-min(obj["z"])
        if towers["dz"][ID]<dz_min: towers.loc[ID]=np.nan
        else:
            towers["z_0"][ID]=np.nanmin(obj["z"])
            towers["z_10"][ID]=np.percentile(obj["z"],10)
            towers["z_25"][ID]=np.percentile(obj["z"],25)
            towers["z_50"][ID]=np.percentile(obj["z"],50)
            towers["z_75"][ID]=np.percentile(obj["z"],75)
            towers["z_90"][ID]=np.percentile(obj["z"],90)
            towers["z_100"][ID]=np.nanmax(obj["z"])
            towers["z_IQR"][ID]=np.percentile(obj["z"],75)-np.percentile(obj["z"],25)
            towers["z_mean"][ID]=np.nanmean(obj["z"])
            
            towers["d_0"][ID]=np.nanmin(obj["diam"])
            towers["d_10"][ID]=np.percentile(obj["diam"],10)
            towers["d_25"][ID]=np.percentile(obj["diam"],25)
            towers["d_50"][ID]=np.percentile(obj["diam"],50)
            towers["d_75"][ID]=np.percentile(obj["diam"],75)
            towers["d_90"][ID]=np.percentile(obj["diam"],90)
            towers["d_100"][ID]=np.nanmax(obj["diam"])
            towers["d_IQR"][ID]=np.percentile(obj["diam"],75)-np.percentile(obj["diam"],25)
            towers["d_mean"][ID]=np.nanmean(obj["diam"])
            
            towers["r_0"][ID]=np.nanmin(obj["dvel"])
            towers["r_10"][ID]=np.percentile(obj["dvel"],10)
            towers["r_25"][ID]=np.percentile(obj["dvel"],25)
            towers["r_50"][ID]=np.percentile(obj["dvel"],50)
            towers["r_75"][ID]=np.percentile(obj["dvel"],75)
            towers["r_90"][ID]=np.percentile(obj["dvel"],90)
            towers["r_100"][ID]=np.nanmax(obj["dvel"])
            towers["r_IQR"][ID]=np.percentile(obj["dvel"],75)-np.percentile(obj["dvel"],25)
            towers["r_mean"][ID]=np.nanmean(obj["dvel"])
            
            towers["v_0"][ID]=np.nanmin(obj["vort"])
            towers["v_10"][ID]=np.percentile(obj["vort"],10)
            towers["v_25"][ID]=np.percentile(obj["vort"],25)
            towers["v_50"][ID]=np.percentile(obj["vort"],50)
            towers["v_75"][ID]=np.percentile(obj["vort"],75)
            towers["v_90"][ID]=np.percentile(obj["vort"],90)
            towers["v_100"][ID]=np.nanmax(obj["vort"])
            towers["v_IQR"][ID]=np.percentile(obj["vort"],75)-np.percentile(obj["vort"],25)
            towers["v_mean"][ID]=np.nanmean(obj["vort"])
            
            towers["rank_0"][ID]=np.nanmin(obj["rank"])
            towers["rank_10"][ID]=np.percentile(obj["rank"],10)
            towers["rank_25"][ID]=np.percentile(obj["rank"],25)
            towers["rank_50"][ID]=np.percentile(obj["rank"],50)
            towers["rank_75"][ID]=np.percentile(obj["rank"],75)
            towers["rank_90"][ID]=np.percentile(obj["rank"],90)
            towers["rank_100"][ID]=np.nanmax(obj["rank"])
            towers["rank_IQR"][ID]=np.percentile(obj["rank"],75)-np.percentile(obj["rank"],25)
            towers["rank_mean"][ID]=np.nanmean(obj["rank"])
            
            towers["size_sum"][ID]=np.sum(obj["size"])
            towers["size_mean"][ID]=np.nanmean(obj["size"])
            towers["vol_sum"][ID]=np.sum(obj["vol"])
            towers["vol_mean"][ID]=np.nanmean(obj["vol"])
            
            towers["x"][ID]=np.average(obj["x"], weights=obj["size"])
            towers["y"][ID]=np.average(obj["y"], weights=obj["size"])
            towers["dz"][ID]=max(obj["z"])-min(obj["z"])
            towers["time"][ID]=time
    towers=towers.dropna()
    print("Towers found: ", len(towers))
    print(towers)
        
    return towers, v_ID


def summarise_rot(tower_list, ID_list):
    """
    Converts list of dataframes per timestep into dataframes sorted by ID, temporal continuity check

    Parameters
    ----------
    tower_list : list
        list of dataframes with rotation detections per timestep.
    ID_list : list
        list of rotation / thunderstorm IDs.

    Returns
    -------
    rotation : pandas dataframe
        rotation metrics per thunderstorm ID.

    """
    delta=datetime.strptime("15","%M")-datetime.strptime("5","%M")
    rotation=pd.DataFrame()
    
    for n in ID_list:
        rot_track=pd.DataFrame(data=None, index=None, columns=["ID", "time", "radar","x", "y", "dz",
                                "A","D","L","P","W","A_range","D_range","L_range","P_range","W_range","A_n","D_n","L_n","P_n","W_n","A_el","D_el","L_el","P_el","W_el",
                                "size_sum","size_mean","vol_sum","vol_mean",
                                "z_0","z_10", "z_25","z_50","z_75","z_90","z_100","z_IQR","z_mean",
                                "r_0","r_10", "r_25","r_50","r_75","r_90","r_100","r_IQR","r_mean",
                                "v_0","v_10", "v_25","v_50","v_75","v_90","v_100","v_IQR","v_mean",
                                "d_0","d_10", "d_25","d_50","d_75","d_90","d_100","d_IQR","d_mean",
                                "rank_0","rank_10", "rank_25","rank_50","rank_75","rank_90","rank_100","rank_IQR","rank_mean",
                                ])
        tl=[]
        for t in range(len(tower_list)):
            mytime=tower_list[t]
            myrot=mytime[mytime.ID==n]
            rot_track=rot_track.append(myrot)
            if len(myrot)>0: tl.append(datetime.strptime(str(myrot.time.astype(int).values[0]), "%y%j%H%M"))
        dt=np.diff(tl)
        close=np.where(dt<=delta)
        ranges=rot_track.A_range.append([rot_track.D_range,rot_track.L_range,rot_track.P_range,rot_track.W_range])
        if len(close[0])<3 or np.nanmax(ranges)<20: continue
        else: rotation=rotation.append(rot_track)
    return rotation

def rot_hist(tower_list, hist):
    hist2=pd.DataFrame(data=None, index=None, columns=["ID","rotdir","cont","dis","last"])
    IDs=np.unique(tower_list.ID)
    fill=np.zeros(len(tower_list))
    tower_list["cont"]=fill
    tower_list["loc"]=fill
    tower_list["flag"]=fill
    time=tower_list.time[0]
    delta=datetime.strptime("25","%M")-datetime.strptime("5","%M")
    for t in range(len(hist)):
        h=hist[t]
        if datetime.strptime(str(time.astype(int).values[0]), "%y%j%H%M")-datetime.strptime(str(h.last.astype(int).values[0]), "%y%j%H%M")>delta: continue
        for n in range(len(IDs)):
            if h.ID==IDs[n]:
                t=tower_list[n]
                rr=np.nanmax([t.A_range,t.D_range,t.L_range,t.P_range,t.W_range])
                h.cont+=1
                if h.cont >= 3: tower_list.iloc[n].cont=1
                h.dis=np.nanmax([rr,h.dis])
                if h.dis >= 20: tower_list.iloc[n].loc=1
                h.last=time
                hist2=hist2.append(h)
    for n in range(len(IDs)):
        if IDs[n] not in hist2.ID:
            t=tower_list[n]
            h=pd.DataFrame(data=None, index=len(hist2), columns=["ID","rotdir","cont","dis","last"])
            h.ID=IDs[n]
            h.cont=0
            h.dis=np.nanmax([t.A_range,t.D_range,t.L_range,t.P_range,t.W_range])
            if h.dis >= 20: tower_list.iloc[n].loc=1
            h.last=time
            hist2=hist2.append(h)
    tower_list.flag=tower_list.cont*tower_list.loc
    return hist2,tower_list