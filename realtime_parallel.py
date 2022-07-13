#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:49:37 2020

@author: mfeldman
"""

import multiprocessing
import numpy as np
import os
import sys
sys.path.append('/users/mfeldman/scripts/ELDES_MESO')
sys.path.append('/scratch/lom/mof/code/ELDES_MESO')
import pandas as pd
pd.options.mode.chained_assignment = None
import skimage.morphology as skim
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore',category=AstropyWarning)
import timeit
import library.variables as variables
import library.io as io
import library.plot as plot
import library.transform as transform
import library.meso as meso
import glob

#%% FUNCTIONS FOR PARALLELIZATION OF ELEVATION PROCESSING
# MUST BE DEFINED IN MAIN SCRIPT

    
def radar_processor(r, radar, cartesian, path, specs, coord, files, shear, resolution,
                    timelist, t, areas, mask, return_dict):
    """
    parallel processing of radars

    Parameters
    ----------
    r : int
        radar number.
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
    return_dict : tuple
        returns result of process, contains dicts of positive and negative rotation of radar.

    """
    manager2 = multiprocessing.Manager()
    return_dict2 = manager2.dict()
    jobs2 = []
    print("Analysing radar: ", radar["radars"][r])
    rotation_pos=variables.meso()
    rotation_neg=variables.meso()
    # PARALLEL ELEVATION PROCESSING
    for el in radar["n_elevations"]-1:
        p2 = multiprocessing.Process(target=elevation_processor, args=(r, el, radar,
                                    cartesian, path, specs, coord, files, shear,
                                    resolution, timelist, t, areas, mask, return_dict2))
        p2.daemon=False
        jobs2.append(p2)
        p2.start()
    # JOIN RESULTS FROM ELEVATIONS
    for proc in jobs2:
        proc.join()
    result2=return_dict2.values()
    for n in range(0,len(result2)):
        s_p, s_n = result2[n]
        rotation_pos["shear_objects"].append(s_p["shear_objects"])
        rotation_pos["prop"]=pd.concat([rotation_pos["prop"],s_p["prop"]], ignore_index=True)
        rotation_pos["shear_ID"].append(s_p["shear_ID"])
        rotation_neg["shear_objects"].append(s_n["shear_objects"])
        rotation_neg["prop"]=pd.concat([rotation_neg["prop"],s_n["prop"]], ignore_index=True)
        rotation_neg["shear_ID"].append(s_n["shear_ID"])
    #RETURN TO MAIN PROGRAM
    return_dict[r]= rotation_pos, rotation_neg
    
def elevation_processor(r, el, radar, cartesian, path, specs, coord, files, shear,
                        resolution, timelist, t, areas, mask, return_dict2):
    """
    Parallel processing of radar elevations

    Parameters
    ----------
    r : int
        radar number.
    el : int
        elevation number.
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
    return_dict2 : tuple
        returns result of process, contains dicts of positive and negative rotation of
        elevation.

    Returns
    -------
    return_dict2 : tuple
        returns result of process, contains dicts of positive and negative rotation of
        elevation.

    """

    print("Analysing sweep: ", radar["elevations"][el])
    # READ VELOCITY DATA
    dvfile=glob.glob(path["temp"]+'DV'+radar["radars"][r]+'/*'+timelist[t]+'*.8'+radar["elevations"][el])[0]
    # dvfile=path["temp"]+'DV'+radar["radars"][r]+'/DV'+radar["radars"][r]+timelist[t] \
    #         +'7L'+specs["sweep_ID_DV"]+radar["elevations"][el]
    myfinaldata, flag1 = io.read_del_data(dvfile)
    #COMPUTE MASK FROM TRT CONTOURS
    p_mask=meso.mask(mask,coord, radar, cartesian, r, el)
    l_mask=meso.mask(areas, coord, radar, cartesian, r, el)
    # exit if too few valid pixels or no velocity data
    if np.nansum(p_mask.flatten())<6: return_dict2[el]=variables.meso(), variables.meso();
    elif flag1 == -1: return_dict2[el]=variables.meso(), variables.meso();
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
                rotation_pos=meso.shear_group(rotation_pos, 1, 
                                                            mfd_conv_m, 
                                                            az_shear_m, 
                                                            ii, 
                                                            resolution, 
                                                            distance, 
                                                            shear, radar,
                                                            radar["elevations"][el], el,
                                                            radar["radars"][r], r,
                                                            cartesian["indices"], timelist[t])
                
                rotation_neg=meso.shear_group(rotation_neg, -1, 
                                                            mfd_conv_m, 
                                                            az_shear_m, 
                                                            ii, 
                                                            resolution, 
                                                            distance, 
                                                            shear, radar,
                                                            radar["elevations"][el], el,
                                                            radar["radars"][r], r,
                                                            cartesian["indices"], timelist[t])
        # return results to radar processing
        return_dict2[el]=rotation_pos, rotation_neg

#%% INITIALIZE PROCESSING
# load case dates and times, load variables, launch timer
time=sys.argv[1]
event=sys.argv[2]
year=sys.argv[3]
radar, cartesian, path, specs, files, shear, resolution=variables.vars(event, year)
coord=variables.read_mask(radar)
#io.makedir(path)


tower_list_p=[]
tower_list_n=[]
#%% PROCESSING CURRENT TIMESTEPS
# launch parallelized rotation detection


t=0
trt_cells, timelist= io.get_TRT(time,path)
t_tic=timeit.default_timer()
#doy=timelist[t][:5]
labels=trt_cells[t]
newlabels=skim.dilation(labels,footprint=np.ones([5,5]))
mask=newlabels>0

t_toc=timeit.default_timer()
print("cell tracking time elapsed [s]: ", t_toc-t_tic)

print("starting rotation detection")
# ROTATION TRACKING
r_tic=timeit.default_timer()

towers_p=pd.DataFrame(columns=["ID", "time", "radar","x", "y", "dz",
                            "A","D","L","P","W","A_range","D_range","L_range",
                            "P_range","W_range","A_n","D_n","L_n","P_n","W_n",
                            "A_el","D_el","L_el","P_el","W_el",
                            "size_sum","size_mean","vol_sum","vol_mean",
                            "z_0","z_10", "z_25","z_50","z_75","z_90","z_100","z_IQR","z_mean",
                            "r_0","r_10", "r_25","r_50","r_75","r_90","r_100","r_IQR","r_mean",
                            "v_0","v_10", "v_25","v_50","v_75","v_90","v_100","v_IQR","v_mean",
                            "d_0","d_10", "d_25","d_50","d_75","d_90","d_100","d_IQR","d_mean",
                            "rank_0","rank_10", "rank_25","rank_50","rank_75","rank_90","rank_100","rank_IQR","rank_mean",
                            ])
towers_n=pd.DataFrame(columns=["ID", "time", "radar","x", "y", "dz",
                            "A","D","L","P","W","A_range","D_range","L_range",
                            "P_range","W_range","A_n","D_n","L_n","P_n","W_n",
                            "A_el","D_el","L_el","P_el","W_el",
                            "size_sum","size_mean","vol_sum","vol_mean",
                            "z_0","z_10", "z_25","z_50","z_75","z_90","z_100","z_IQR","z_mean",
                            "r_0","r_10", "r_25","r_50","r_75","r_90","r_100","r_IQR","r_mean",
                            "v_0","v_10", "v_25","v_50","v_75","v_90","v_100","v_IQR","v_mean",
                            "d_0","d_10", "d_25","d_50","d_75","d_90","d_100","d_IQR","d_mean",
                            "rank_0","rank_10", "rank_25","rank_50","rank_75","rank_90","rank_100","rank_IQR","rank_mean",
                            ])
rotation_pos=variables.meso(); rotation_neg=variables.meso()
print("Analysing timestep: ", timelist[t])
# PARALLEL RADAR PROCESSING
if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
io.blockPrint()
for r in radar["n_radars"]:
    p = multiprocessing.Process(target=radar_processor, args=(r, radar, cartesian,
                              path, specs, coord, files, shear, resolution, timelist,
                              t, newlabels, mask, return_dict))
    jobs.append(p)
    p.start()
for proc in jobs:
    proc.join()
# JOIN RESULTS FROM RADARS
result=return_dict.values()
io.enablePrint()
for n in range(0,len(result)):
    s_p, s_n = result[n]
    rotation_pos["shear_objects"].append(s_p["shear_objects"])
    rotation_pos["prop"]=pd.concat([rotation_pos["prop"],s_p["prop"]], ignore_index=True)
    rotation_pos["shear_ID"].append(s_p["shear_ID"])
    rotation_neg["shear_objects"].append(s_n["shear_objects"])
    rotation_neg["prop"]=pd.concat([rotation_neg["prop"],s_n["prop"]], ignore_index=True)
    rotation_neg["shear_ID"].append(s_n["shear_ID"])
# MERGE OBJECT DETECTION FROM RADARS
vert_p, v_ID_p = meso.tower(rotation_pos, newlabels, radar, shear, r, timelist[t], path)
vert_n, v_ID_n = meso.tower(rotation_neg, newlabels, radar, shear, r, timelist[t], path)

tower_list_p.append(vert_p)
tower_list_n.append(vert_n)
r_toc=timeit.default_timer()
print("time elapsed [s]: ", r_toc-r_tic)
t_toc=timeit.default_timer()
print("Computation time timestep: [s] ",t_toc-t_tic)
phist,nhist=io.read_histfile(path)
phist,vert_p=meso.rot_hist(vert_p, phist,time)
nhist,vert_n=meso.rot_hist(vert_n, nhist,time)
io.write_histfile(phist,nhist,path)
pfile=path["temp"]+'ROT/'+'PROT'+str(year)+str(event)+str(time)+'.json'
io.write_geojson(vert_p,pfile)
nfile=path["temp"]+'ROT/'+'NROT'+str(year)+str(event)+str(time)+'.json'
io.write_geojson(vert_n,nfile)

