#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:06:13 2020

@author: feldmann
"""

# import pyart
import numpy as np
import pandas as pd
import glob
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import copy
from datetime import datetime
from datetime import timedelta
import library.io as io
import library.variables as variables
import library.transform as transform
#%%
def mask(mask, coord, radar, cartesian, r, el):
    """
    converts Cartesian thunderstorm mask to polar grid using lookup-table

    Parameters
    ----------
    mask : 2D array
        Cartesian grid of thunderstorm cells.
    coord : list
        list of coordinate conversions (look-up-table).
    radar : dict
        contains radar information.
    cartesian : dict
        contains coordinate information.
    r : int
        radar number.
    el : int
        elevation number.

    Returns
    -------
    p_mask : 2D array
        polar conversion for the given radar and elevation of the thunderstorm array.

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

def pattern_vectors(shear, min_shear_far, min_shear_near, min_length, distance):
    ##Identification of 1D pattern vectors
    ##continuous elements with certain azimuthal length
    
    print("Identifying pattern vectors")
    shear_ID=np.zeros(shear.shape)
    shearID=np.zeros(shear.shape);# shearID[:]=np.nan
    IDpos=1
    for m in range(0,shear.shape[1]):
        #print (m)
        for n in range(0,shear.shape[0]-2):
            if n<=40: shear_thresh=min_shear_near
            if n<200 and n>40: shear_thresh=min_shear_near-(min_shear_near-min_shear_far)*((n-40)/160)
            if n>=200: shear_thresh=min_shear_far
            # print(n, shear_thresh)
            if all(shear[n:n+3,m]>shear_thresh): shear_ID[n:n+3,m]=IDpos; # and all(myshear_cor[n:n+3,m]<3)
            elif all(shear[n-1:n+2,m]>shear_thresh) and shear[n+2,m]<=shear_thresh: IDpos+=1
            
    for n in range(0,IDpos+1):
        indices=np.where(shear_ID==n)
        if np.sum(distance[indices])<min_length:
            shear_ID[shear_ID==n]=0
            
    shearID[shear_ID>0]=1
    
    return shearID

def neighbor_check(n,m,shearID,shear_groups,g_ID,ind):
    ##recursive function merging positive objects
    ##Now replace with scikit-image function
    for a in range(n-1,n+2):
        if a>359:a=a-360
        for b in range(m-2,m+3):
            if b>shearID.shape[1]-1:b=shearID.shape[1]-1
            if shearID[a,b]>0 and shear_groups[a,b]==0:
                ind=1
                shear_groups[a,b]=g_ID
                shear_groups,ind=neighbor_check(a,b,shearID,shear_groups,g_ID,ind)
                
    return shear_groups,ind


def shear_group(rotation, sign, myfinaldata, az_shear, labels, resolution, distance, shear, radar, EL, el, R, r, coord, time):
    ##Identification of 2D rotation patches
    min_width=shear["width"]
    az_shear=sign*az_shear
    myfinaldata=sign*myfinaldata
    ##Obtain 1D pattern vectors
    shearID=pattern_vectors(az_shear, shear["far1"], 
                                            shear["near1"], 
                                            shear["length"], 
                                            distance)
    rotation["pattern_vector"]=shearID
    ##Merge 1D pattern vectors to 2D patches
    print("Identifying shear areas")
    shear_groups, g_ID = ndi.label(shearID)
    shear_prop=[]
    for n in range(1,g_ID+1):
        ##Establish range-dependent thresholds
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
            if cen_r>=200:
                min_shear_2=shear["near2"]-shear["near2d"]
                min_rvel=shear["rvel1"]-shear["rvel1d"]
                min_vort=shear["vort1"]-shear["vort1d"]
            ##Compute strength metrics for 2D objects
            maxshear=max(az_shear[indices],key=abs)
            binary=np.zeros(myfinaldata.shape)
            binary[:]=np.nan
            binary[indices]=1
            filt=binary*myfinaldata
            ma=np.nanmax(filt); mi=np.nanmin(filt)
            dvel=abs(ma-mi)/2
            lmax=np.where(filt==ma); lmin=np.where(filt==mi)
            xmax,ymax,zmax=coord[:,lmax[0],lmax[1]]
            xmin,ymin,zmin=coord[:,lmin[0],lmin[1]]
            dis=np.array([])
            for n2 in range(len(xmax)):
                dx=xmin-xmax[n2]; dy=ymin-ymax[n2]
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
            ## If any of the thresholds not met, discard 2D patch
            if maxlen<min_width or ratio<(1/3) or ratio>3 or vort<min_vort or dvel<min_rvel or mindis<1000 or mindis>10000:#or abs(maxshear)<min_shear_2  or av_ref<20
                shear_groups[shear_groups==n]=0
            else:
                print("rotation characteristics:", dvel, vort, dis)
                x=[]; y=[]; z=[];
                x, y, z = coord[:,indices[0],indices[1]]
                cart_centroids = np.mean(x), np.mean(y), np.mean(z)
                shear_prop.append([n, time, el, R, indices, cart_centroids[0]+radar["x"][r], cart_centroids[1]+radar["y"][r], cart_centroids[2]+radar["z"][r], dvel, vort, mindis, rank, vertical_ID, size, vol, cen_r])
                    
    #sheargroups=np.zeros(myfinaldata.shape); sheargroups[:]=np.nan
    #sheargroups[shear_groups>0]=1
    ##Collect all 2D patches
    shear_prop=pd.DataFrame(data=shear_prop, columns=["ID",  "time", "elevation", "radar", "indices", "x", "y", "z", "dvel", "vort", "diam", "rank", "v_ID", "size", "vol", "range"])
    #"trtlat", "trtlon",
    print(shear_prop)
    print("Identified shear areas: ", len(shear_prop))

    rotation["prop"]=pd.concat([rotation["prop"],shear_prop], ignore_index=True)
    
    
    return rotation


def tower(rotation, areas, radar, shear, time, path):
    ##merging objects within same thunderstorm
    print("checking for vertical towers")
    ##If too few 2D patches, discard
    if len(rotation["prop"])<2:
        print("too few 2D patches")
        towers=variables.rot_df()
        v_ID=0
        return towers, v_ID
    
    prop=rotation["prop"].copy()
    v_ID=prop["v_ID"].values
    print("building 3D towers from 2D objects")
    n=np.unique(prop["v_ID"].values)
    print(n)
    headers=variables.rot_df().columns
    towers=pd.DataFrame(data=0.0, index=n, columns=headers)
    ## Build rotation tower per thunderstorm ID
    for ID in n:
        obj=prop.where(prop["v_ID"]==ID).dropna()
        if len(obj)<1: continue
        towers["ID"][ID]=ID
        #towers["trtlat"][ID]=obj["trtlat"].values
        #towers["trtlon"][ID]=obj["trtlon"].values
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
        # Identify minimum range from any detecting radar
        # Establish range-dependent depth threshold
        ra=np.nanmin([towers.A_range,towers.D_range,towers.L_range,towers.P_range,towers.W_range])
        if ra>100:
            dz_min=shear["zu"]-shear["zd"]
        if ra>20 and ra<=100:
            dz_min=shear["zu"]-shear["zd"]*((ra-20)/80)
        if ra<=20:
            dz_min=shear["zu"]-shear["zu"]*((20-ra)/20)
        print("Minimum range, depth threshold", ra,dz_min)
        
        towers["dz"][ID]=max(obj["z"])-min(obj["z"])
        # If depth threshold not met, discard 3D object
        if towers["dz"][ID]<dz_min: towers.loc[ID]=np.nan; print('shear area too shallow', ID); continue
        
        # All criteria met, fill rotation-tower dataframe with percentiles of 2D patches
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
        print('Object merged; depth, rank, vorticity and rvel: ',towers["dz"][ID],towers["rank_90"][ID],towers["v_90"][ID],towers["r_90"][ID])
    towers=towers.dropna()
    print("Towers found: ", len(towers))
    print(towers)
        
    return towers, v_ID

def drop_duplicates(towers):
    ##merging duplicate objects after merging radars
    # no longer used, was used while mesocyclones were first established per radar and then combined
    for n_obj in range(0,towers.shape[0]):
        #row = towers.iloc[n_obj]
        dx=np.abs(towers["centroid_cart"][n_obj][0] - towers["centroid_cart"][:][0])
        dy=np.abs(towers["centroid_cart"][n_obj][1] - towers["centroid_cart"][:][1])
        distance = np.sqrt(dx*dx + dy*dy)
        equal = np.where(distance < 2)
        towers["radar"][n_obj]=[towers["radar"][equal]]
        towers["rvel_max"][n_obj]=max(np.array(towers["rvel_max"][equal]))
        towers["shearsum_max"][n_obj]=max(np.array(towers["shearsum_max"][equal]))
        towers["depth"][n_obj]=max(np.array(towers["depth"][equal]))
        towers.drop(n_obj, axis=0)
    return towers

def summarise_rot(tower_list, ID_list):
    ##previous function to check for rotation consistency, not used in realtime
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
            rot_track=pd.concat((rot_track,myrot),axis=1)#rot_track.append(myrot)
            if len(myrot)>0: tl.append(datetime.strptime(str(myrot.time.astype(int).values[0]), "%y%j%H%M"))
        dt=np.diff(tl)
        close=np.where(dt<=delta)
        # if len(rot_track)<2 or all(dt>delta):continue
        # if len(rot_track)<1: continue
        ranges=pd.concat((rot_track.A_range,rot_track.D_range,rot_track.L_range,rot_track.P_range,rot_track.W_range),axis=1)
        # ranges=pd.DataFrame.from_records(rot_track.range.to_numpy())
        if len(close[0])<3 or np.nanmax(ranges)<20: continue
        else: rotation=pd.concat((rotation,rot_track),axis=1)#rotation.append(rot_track)
    return rotation

def rot_dist(tower_list):
    """
    Check mesocyclone for sufficient distance from radars

    Parameters
    ----------
    tower_list : list of dataframes
        list of rotation dataframes.

    Returns
    -------

    tower_list : list of dataframes
        list of rotation dataframes with continuity flags.

    """
    IDs=np.unique(tower_list.ID)
    fill=np.zeros(len(tower_list)); fill[:]=-1
    tower_list["cont"]=fill
    tower_list["dist"]=fill
    tower_list["flag"]=fill
    #Add flag information to dataframes
    for n in range(len(IDs)):
        t=tower_list.iloc[n]
        dist=np.nanmax([t.A_range,t.D_range,t.L_range,t.P_range,t.W_range])
        if dist >= 20: tower_list.dist.iat[n]=1
    return tower_list

def rot_hist(tower_list, hist,time):
    """
    Check mesocyclone history for time continuity

    Parameters
    ----------
    tower_list : list of dataframes
        list of rotation dataframes.
    hist : dataframe
        dataframe with rotation history.
    time : string
        timestep.

    Returns
    -------
    hist2 : dataframe
        dataframe with new rotation history.
    tower_list : list of dataframes
        list of rotation dataframes with continuity flags.

    """
    #generate new history dataframe
    hist2=pd.DataFrame(data=None, index=None, columns=["ID","cont","dist","latest"])
    IDs=np.unique(tower_list.ID)
    #add flags to rotation dataframe
    fill=np.zeros(len(tower_list))
    tower_list["cont"]=fill
    tower_list["dist"]=fill
    tower_list["flag"]=fill
    #establish minimum time criterion (if last detection more than 20 min ago, stop counting continuity)
    delta=datetime.strptime("25","%M")-datetime.strptime("5","%M")
    for t in range(len(hist)):
        h=hist.iloc[t]
        a=0
        # discard time continuity for detections more than 20 min ago
        if datetime.strptime(str(time), "%y%j%H%M")-datetime.strptime(str(int(h.latest)), "%y%j%H%M")>delta: continue
        for n in range(len(IDs)):
            # print(h.ID,IDs[n])
            if h.ID==IDs[n]:
                a=1
                t=tower_list.iloc[n]
                rr=np.nanmax([t.A_range,t.D_range,t.L_range,t.P_range,t.W_range])
                #add detection to counter, once more than 3 detections are reached, time criterion fulfilled
                h.cont+=1
                if h.cont >= 3: tower_list.cont.iat[n]=1
                #check range from furthest radar, if >20, distance criterion fulfilled
                h.dist=np.nanmax([rr,h.dist])
                if h.dist >= 20: tower_list.dist.iat[n]=1
                h.latest=int(time)
                hh=pd.DataFrame(h).transpose()
                hist2=pd.concat((hist2,hh),axis=0)
        #if history and rotation empty, return empty dataframe
        if a==0:
            hh=pd.DataFrame(h).transpose()
            hist2=pd.concat((hist2,hh),axis=0)
    #Add flag information to dataframes
    for n in range(len(IDs)):
        if IDs[n] not in list(hist2.ID):
            t=tower_list.iloc[n]
            h=pd.DataFrame(data=None, index=[len(hist2)], columns=["ID","cont","dist","latest"])
            h.ID=IDs[n]
            h.cont=0
            h.dist=np.nanmax([t.A_range,t.D_range,t.L_range,t.P_range,t.W_range])
            if h.dist.values[0] >= 20: tower_list.dist.iat[n]=1
            h.latest=int(time)
            # h=pd.DataFrame(h).transpose()
            hist2=pd.concat((hist2,h),axis=0)#hist2.append(h) #pd.concat([hist2,h])
    tower_list.flag=tower_list.cont*tower_list.dist
    return hist2,tower_list


def cell_loop(ii, l_mask, az_shear, mfd_conv, rotation_pos, rotation_neg, distance, resolution, shear, radar, coord, timelist, r, el):
    """
    Launch rotation detection within thunderstorm cells

    Parameters
    ----------
    ii : int
        thunderstorm cell ID.
    cellvar : tuple
        contains necessary variables.

    Returns
    -------
    rotation_pos : dict
        positive rotation detections.
    rotation_neg : dict
        negative rotation detections.

    """
    
    #l_mask, az_shear, mfd_conv, rotation_pos, rotation_neg, distance, resolution, shear, radar, coord, timelist, r, el= cellvar
    t=0
    #mask data to desired thunderstorm cell
    binary=l_mask==ii
    az_shear_m=az_shear*binary
    mfd_conv_m=mfd_conv*binary
    if np.nanmax(abs(az_shear_m.flatten('C')))>=3:
        print("Identifying rotation shears")
        # rotation object detection for both signs
        rotation_pos=shear_group(rotation_pos, 1, 
                                                    mfd_conv_m, 
                                                    az_shear_m, 
                                                    ii,
                                                    resolution, 
                                                    distance, 
                                                    shear, radar,
                                                    radar["elevations"][el], el,
                                                    radar["radars"][r], r,
                                                    coord, timelist[t])
        
        rotation_neg=shear_group(rotation_neg, -1, 
                                                    mfd_conv_m, 
                                                    az_shear_m, 
                                                    ii,
                                                    resolution, 
                                                    distance, 
                                                    shear, radar,
                                                    radar["elevations"][el], el,
                                                    radar["radars"][r], r,
                                                    coord, timelist[t])
    
    return rotation_pos, rotation_neg
