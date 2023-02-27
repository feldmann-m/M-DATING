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
    for a in range(n-1,n+2):
        if a>359:a=a-360
        for b in range(m-2,m+3):
            if b>shearID.shape[1]-1:b=shearID.shape[1]-1
            if shearID[a,b]>0 and shear_groups[a,b]==0:
                ind=1
                shear_groups[a,b]=g_ID
                shear_groups,ind=neighbor_check(a,b,shearID,shear_groups,g_ID,ind)
                
    return shear_groups,ind

#@profile
def shear_group(rotation, sign, myfinaldata, az_shear, labels, resolution, distance, shear, radar, EL, el, R, r, coord, time):
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
    # shear_groups=np.zeros(shearID.shape);
    # g_ID=1;
    # for n in range(1,shearID.shape[0]-1):
    #     for m in range(2,shearID.shape[1]-2):
    #         shear_groups,indp=neighbor_check(n,m,shearID,shear_groups,g_ID,0)
    #         if indp==1:g_ID+=1
    shear_groups, g_ID = ndi.label(shearID)
    shear_prop=[]
    for n in range(1,g_ID+1):
        indices=np.where(shear_groups==n)
        if len(np.unique(indices[1]))>2:
            vertical_ID=labels#np.unique(labels[indices])[0]
            size=np.nansum(distance[indices]*resolution)
            vol=np.nansum(distance[indices]*resolution*resolution)
            # av_ref=np.nanmean(myref[indices])
            cen_r=np.mean(indices[1])
            # centroids=cen_az, cen_r
            if cen_r<=40: #min_shear_2=shear["near2"]; min_rvel=shear["rvel1"]; min_vort=shear["vort1"]
                min_shear_2=shear["near2"]-shear["near2d"]*((40-cen_r)/40)
                min_rvel=shear["rvel1"]-shear["rvel1d"]*((40-cen_r)/40)
                min_vort=shear["vort1"]-shear["vort1d"]*((40-cen_r)/40)
            if cen_r<200 and cen_r>40:
                min_shear_2=shear["near2"]-shear["near2d"]*((cen_r-40)/160)
                min_rvel=shear["rvel1"]-shear["rvel1d"]*((cen_r-40)/160)
                min_vort=shear["vort1"]-shear["vort1d"]*((cen_r-40)/160)
            if cen_r>=200: min_shear_2=shear["near2"]-shear["near2d"]; min_rvel=shear["rvel1"]-shear["rvel1d"]; min_vort=shear["vort1"]-shear["vort1d"]
            # print(cen_r,min_shear_2,min_rvel,min_vort)
            # magnitude_2=np.abs(np.sum(az_shear[indices]))*resolution
            # print(n,g_ID,cen_r,min_vort,min_rvel)
            maxshear=max(az_shear[indices],key=abs)
            binary=np.zeros(myfinaldata.shape)
            binary[:]=np.nan
            binary[indices]=1
            filt=binary*myfinaldata
            ma=np.nanmax(filt); mi=np.nanmin(filt)
            dvel=abs(ma-mi)/2
            lmax=np.where(filt==ma); lmin=np.where(filt==mi)
            # mmax=np.nanmean(lmax[0]), np.nanmean(lmax[1])
            # mmin=np.nanmean(lmin[0]), np.nanmean(lmin[1])
            # xmax,ymax,zmax=pyart.core.antenna_to_cartesian(lmax[1], lmax[0], el)
            # xmin,ymin,zmin=pyart.core.antenna_to_cartesian(lmin[1], lmin[0], el)
            xmax,ymax,zmax=coord[:,lmax[0],lmax[1]]
            xmin,ymin,zmin=coord[:,lmin[0],lmin[1]]
            dis=np.array([])
            for n in range(len(xmax)):
                dx=xmin-xmax[n]; dy=ymin-ymax[n]
                dis=np.append(dis,np.sqrt(dx*dx + dy*dy),axis=0)
            mindis=np.nanmin(dis)
            vort=4*dvel/mindis
            if np.isnan(dvel): dvel=0
            vec_width=[]
            for m in np.unique(indices[1]):
                #if m >359: m=359
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
                x, y, z = coord[:,indices[0],indices[1]]
                # for n in range(len(indices[0])):
                #     #print(r, n, indices[1][n], indices[0][n])
                #     cart_indices=pyart.core.antenna_to_cartesian(indices[1][n]/2, indices[0][n], el)
                #     #cart_indices=fg_indices[r][indices[1][n], indices[0][n], el]
                #     x.append(cart_indices[0])
                #     y.append(cart_indices[1])
                #     z.append(cart_indices[2])
                cart_centroids = np.mean(x), np.mean(y), np.mean(z)
                shear_prop.append([n, time, el, R, indices, cart_centroids[0]+radar["x"][r], cart_centroids[1]+radar["y"][r], cart_centroids[2]+radar["z"][r], dvel, vort, mindis, rank, vertical_ID, size, vol, cen_r])
                    
    sheargroups=np.zeros(myfinaldata.shape); sheargroups[:]=np.nan
    sheargroups[shear_groups>0]=1
    shear_prop=pd.DataFrame(data=shear_prop, columns=["ID", "time", "elevation", "radar", "indices", "x", "y", "z", "dvel", "vort", "diam", "rank", "v_ID", "size", "vol", "range"])
    print(shear_prop)
    print("Identified shear areas: ", len(shear_prop))
    # if len(rotation["prop"]) and rotation["prop"].elevation.iloc[-1]==el:
    #     rotation["shear_objects"][-1]=np.nansum([rotation["shear_objects"][-1],sheargroups],axis=0)
    #     rotation["shear_ID"][-1]=np.nansum([rotation["shear_ID"][-1],shear_groups],axis=0)
    # else:
    # rotation["shear_ID"].append(shear_groups)
    # rotation["shear_objects"].append(sheargroups)
    rotation["prop"]=pd.concat([rotation["prop"],shear_prop], ignore_index=True)
    
    
    return rotation

def prox_rec(N, n, prop, p_ID):
    print("entering recursion")
    for M in range(N+1,n):
        dx=np.abs(prop.x[N] - prop.x[M])
        dy=np.abs(prop.y[N] - prop.y[M])
        distance=np.sqrt(dx*dx + dy*dy)
        if distance<5000:
            #print(distance, p_ID)
            prop.v_ID[N]=p_ID
            prop.v_ID[M]=p_ID
            if M<n-1:
                prox_rec(M, n, prop, p_ID)
    return prop

def proximity_check(rotation, radar):
    ##general x, y proximity check between shear areas
    print("checking for proximity")
    p_ID=1
    prop=rotation["prop"].copy()
    n=len(prop)
    for N in range(n):
        if prop.v_ID[N]<1:
            prop = prox_rec(N, n, prop, p_ID)
            p_ID+=1
                    
    return prop, p_ID

def cell_assignment(rotation, areas, radar,r):
    print("checking for cells")
    prop=rotation["prop"].copy()
    n=len(prop)
    for N in range(n):
        x=int((prop.x[N]+radar["x"][r]-254000)/1000)
        y=int((prop.y[N]+radar["y"][r]+159000)/1000)
        c_ID=areas[y,x]
        prop.v_ID[N]=c_ID
        # print(c_ID)
                    
    return prop, c_ID

def tower(rotation, areas, radar, shear, r, time, path):
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
    # prop, v_ID=cell_assignment(rotation, areas, radar,r)
    
    prop=rotation["prop"].copy()
    v_ID=prop["v_ID"].values
    print("building towers")
    # n=int(max(prop["v_ID"].values)+1)
    # m=int(min(prop["v_ID"].values)+1)
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
        # tow=obj.drop(columns=['x','y','z','weight','ID'])
        # name=path["images"]+str(time)+'_'+str(ID)+radar["radars"][r]+'.txt'
        # io.write_track(tow, name)
        towers["ID"][ID]=ID
        towers["radar"][ID]=np.unique(obj["radar"].values) #obj["radar"].values.tolist()
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
        # towers.range[ID]=np.average(obj["range"], weights=obj["size"])*0.5
        # towers["n_objects"][ID]=len(obj["vol"])
        # towers["n_elev"][ID]=len(np.unique(obj["elevation"]))
        
        # towers["range"][ID]=r_range
        # towers["n_objects"][ID]=r_n
        # towers["n_elev"][ID]=r_elev
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
        # if np.max(obj["dvel"])< rot_max or np.max(obj["vort"])<vort_max: towers["r_100"][ID]=np.nan; continue
        towers["dz"][ID]=max(obj["z"])-min(obj["z"])
        if towers["dz"][ID]<dz_min: towers.loc[ID]=np.nan; continue
        
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
        
        #towers["shearsum_max"][ID]=np.max(obj["shearsum"])
        towers["x"][ID]=np.average(obj["x"], weights=obj["size"])#+radar["x"][r]
        towers["y"][ID]=np.average(obj["y"], weights=obj["size"])#+radar["y"][r]
        towers["dz"][ID]=max(obj["z"])-min(obj["z"])
        # if towers["dz"][ID]<dz_min: towers["dz"][ID]=np.nan
        towers["time"][ID]=time
    towers=towers.dropna()
    print("Towers found: ", len(towers))
    print(towers)
        
    return towers, v_ID

def drop_duplicates(towers):
    ##merging duplicate objects after merging radars
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
        # if len(rot_track)<2 or all(dt>delta):continue
        # if len(rot_track)<1: continue
        ranges=rot_track.A_range.append([rot_track.D_range,rot_track.L_range,rot_track.P_range,rot_track.W_range])
        # ranges=pd.DataFrame.from_records(rot_track.range.to_numpy())
        if len(close[0])<3 or np.nanmax(ranges)<20: continue
        else: rotation=rotation.append(rot_track)
    return rotation

def rot_hist(tower_list, hist,time):
    hist2=pd.DataFrame(data=None, index=None, columns=["ID","cont","dist","latest"])
    IDs=np.unique(tower_list.ID)
    fill=np.zeros(len(tower_list))
    tower_list["cont"]=fill
    tower_list["dist"]=fill
    tower_list["flag"]=fill
    # time=tower_list.time.iloc[0]
    delta=datetime.strptime("25","%M")-datetime.strptime("5","%M")
    for t in range(len(hist)):
        h=hist.iloc[t]
        a=0
        if datetime.strptime(str(time), "%y%j%H%M")-datetime.strptime(str(int(h.latest)), "%y%j%H%M")>delta: continue
        for n in range(len(IDs)):
            if h.ID==IDs[n]:
                a=1
                t=tower_list.iloc[n]
                rr=np.nanmax([t.A_range,t.D_range,t.L_range,t.P_range,t.W_range])
                h.cont+=1
                if h.cont >= 3: tower_list.cont.iat[n]=1
                h.dist=np.nanmax([rr,h.dist])
                if h.dist >= 20: tower_list.dist.iat[n]=1
                h.latest=int(time)
                hist2=hist2.append(h) #pd.concat([hist2,h])
        if a==0:
            hist2=hist2.append(h) #pd.concat([hist2,h])
    for n in range(len(IDs)):
        if IDs[n] not in list(hist2.ID):
            t=tower_list.iloc[n]
            h=pd.DataFrame(data=None, index=[len(hist2)], columns=["ID","cont","dist","latest"])
            h.ID=IDs[n]
            h.cont=0
            h.dist=np.nanmax([t.A_range,t.D_range,t.L_range,t.P_range,t.W_range])
            if h.dist.values[0] >= 20: tower_list.dist.iat[n]=1
            h.latest=int(time)
            hist2=hist2.append(h) #pd.concat([hist2,h])
    tower_list.flag=tower_list.cont*tower_list.dist
    return hist2,tower_list

def proc_el(r, el, radar, cartesian, path, specs, coord, files, shear, resolution, timelist, t, areas, mask):
    """
    

    Parameters
    ----------
    r : TYPE
        DESCRIPTION.
    el : TYPE
        DESCRIPTION.
    radar : dict
        variable containing all radar information (see library.variables.py).
    cartesian : dict
        variable containing information of Cartesian grid.
    path : dict
        variable containing all data and saving paths.
    specs : dict
        variable containing setup specs.
    coord : list
        radar-relative Cartesian coordinates of polar grid.
    files : dict
        list of velocity files (currently unused).
    shear : dict
        variable containing all thresholds.
    resolution : float
        radial resolution in km.
    timelist : list
        list with all processed timesteps.
    t : int
        number of current timestep.
    areas : 2D array
        Cartesian grid with spatial grid of thunderstorm IDs.
    mask : 2D array
        Cartesian binary grid corresponding to thunderstorms.
    return_dict : tuple
        returns result of process, contains dicts of positive and negative rotation of radar.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    
    
    rotation_pos=variables.meso()
    rotation_neg=variables.meso()
    
    
    dvfile=glob.glob(path["dvdata"]+'DV'+radar["radars"][r]+'/*'+timelist[t]+'*.8'+radar["elevations"][el])[0]
    # dvfile=path["temp"]+'DV'+radar["radars"][r]+'/DV'+radar["radars"][r]+timelist[t] \
    #         +'7L'+specs["sweep_ID_DV"]+radar["elevations"][el]
    myfinaldata, flag1 = io.read_del_data(dvfile)
    #COMPUTE MASK FROM TRT CONTOURS
    print(r, el)
    p_mask=mask(mask,coord, radar, cartesian, r, el)
    l_mask=mask(areas, coord, radar, cartesian, r, el)
    # exit if too few valid pixels or no velocity data
    if np.nansum(p_mask.flatten())<6: 
        return variables.meso(), variables.meso();
    elif flag1 == -1: 
        return variables.meso(), variables.meso();
    else:
        # derive azimuthal shear
        nyquist=radar["nyquist"][el]
        mfd_conv=transform.conv(myfinaldata)
        distance=variables.distance(myfinaldata, resolution)
        mfd_conv[:,40:]=myfinaldata[:,40:]
        az_shear = transform.az_cd(mfd_conv, nyquist, 0.8*nyquist, resolution, 2)[0]
        rotation_pos=variables.meso(); rotation_neg=variables.meso()
        ids=np.unique(l_mask)
        ids=ids[ids>0]
        for ii in ids:
            # mask data per thunderstorm cell
            binary=l_mask==ii
            az_shear_m=az_shear*binary
            mfd_conv_m=mfd_conv*binary
            if np.nanmax(abs(az_shear_m.flatten('C')))>=3:
                print("Identifying anticyclonic shears")
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
                                                            cartesian["indices"], timelist[t])
                
                rotation_neg=shear_group(rotation_neg, -1, 
                                                            mfd_conv_m, 
                                                            az_shear_m, 
                                                            ii, 
                                                            resolution, 
                                                            distance, 
                                                            shear, radar,
                                                            radar["elevations"][el], el,
                                                            radar["radars"][r], r,
                                                            cartesian["indices"], timelist[t])
    
    return rotation_pos, rotation_neg
