#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:23:03 2020

@author: mfeldman
"""

#%%
import pyart
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
sys.path.append('/users/mfeldman/scripts/mesocyclone_detection')
sys.path.append('/home/feldmann/Documents/Research/CSCS_downloads/py_scripts/rotation')
import copy
import timeit
from datetime import datetime
from datetime import timedelta
import library.variables as variables
import library.io as io
import library.plot as plot
import library.transform as transform
import library.meso as meso
import library.cells as cells
import library.aggregate as aggregate
import skimage.measure as skime
import skimage.morphology as skim
import shapefile
import nmmn.plots
turbo=nmmn.plots.turbocmap()
#%%
borders = shapefile.Reader('/users/mfeldman/map_radar/Border_CH.shp')
#%%
year='2017'
event="17214"
event_begin="192321615"#yydddhhmm"
event_end="192321900"#"yydddhhmm"
c_tic=timeit.default_timer()
cases=variables.case()
#%%
radar, cartesian, path, specs, files, shear, resolution=variables.vars(event, year)
coord=variables.mask_coord(radar)

#%%
trt_cells, timelist= io.TRT_to_grid(year, event, path)

if len(timelist)>0:io.unzip(path,int(event),year)
n_time=np.arange(0,len(timelist))

tower_list_p=[]
tower_list_n=[]
#%%
path["images"]='/scratch/mfeldman/case_manuscript/'
specs["test_ID"]='_clipped'
io.makedir(path)
#%%
for t in n_time[2:5]:
    t_tic=timeit.default_timer()
    doy=str(timelist[t])[:5]
    labels=trt_cells[t,:,:]
    newlabels=skim.dilation(labels,selem=np.ones([5,5]))
    mask=newlabels>0

    t_toc=timeit.default_timer()
    print("cell tracking time elapsed [s]: ", t_toc-t_tic)

    print("starting rotation detection")
    ## %% ROTATION TRACKING
    r_tic=timeit.default_timer()
    towers_p=pd.DataFrame(columns=["ID", "time", "radar","x", "y", "dz",
                                "A","D","L","P","W","A_range","D_range","L_range","P_range","W_range","A_n","D_n","L_n","P_n","W_n","A_el","D_el","L_el","P_el","W_el",
                                "size_sum","size_mean","vol_sum","vol_mean",
                                "z_0","z_10", "z_25","z_50","z_75","z_90","z_100","z_IQR","z_mean",
                                "r_0","r_10", "r_25","r_50","r_75","r_90","r_100","r_IQR","r_mean",
                                "v_0","v_10", "v_25","v_50","v_75","v_90","v_100","v_IQR","v_mean",
                                "d_0","d_10", "d_25","d_50","d_75","d_90","d_100","d_IQR","d_mean",
                                "rank_0","rank_10", "rank_25","rank_50","rank_75","rank_90","rank_100","rank_IQR","rank_mean",
                                ])
    towers_n=pd.DataFrame(columns=["ID", "time", "radar","x", "y", "dz",
                                "A","D","L","P","W","A_range","D_range","L_range","P_range","W_range","A_n","D_n","L_n","P_n","W_n","A_el","D_el","L_el","P_el","W_el",
                                "size_sum","size_mean","vol_sum","vol_mean",
                                "z_0","z_10", "z_25","z_50","z_75","z_90","z_100","z_IQR","z_mean",
                                "r_0","r_10", "r_25","r_50","r_75","r_90","r_100","r_IQR","r_mean",
                                "v_0","v_10", "v_25","v_50","v_75","v_90","v_100","v_IQR","v_mean",
                                "d_0","d_10", "d_25","d_50","d_75","d_90","d_100","d_IQR","d_mean",
                                "rank_0","rank_10", "rank_25","rank_50","rank_75","rank_90","rank_100","rank_IQR","rank_mean",
                                ])
    rotation_pos=variables.meso()
    rotation_neg=variables.meso()
    print("Analysing timestep: ", timelist[t])
    for r in radar["n_radars"][0:1]:
        print("Analysing radar: ", radar["radars"][r])
        for el in radar["n_elevations"]-1:
            print("Analysing sweep: ", radar["elevations"][el])
            dvfile=path["temp"]+'DV'+radar["radars"][r]+'/DV'+radar["radars"][r]+str(timelist[t])+'7L'+specs["sweep_ID_DV"]+radar["elevations"][el]
            mlfile=path["temp"]+'/ML'+radar["radars"][r]+str(timelist[t])+'0U'+specs["sweep_ID_ML"]+radar["elevations"][el]
            time=variables.time(dvfile)
        
            myfinaldata, flag1 = io.read_del_data(dvfile)
        
            myvel, myref, flag2 = io.read_raw_data(mlfile)
            # if flag1 == -1: myfinaldata=myvel; print('USING RAW VELOCITY')
            # if flag2 == -1: continue
            # else:
            nyquist=radar["nyquist"][el]
            
            mfd_conv=transform.conv(myfinaldata)
            distance=variables.distance(myfinaldata, resolution)
            mfd_conv[:,40:]=myfinaldata[:,40:]
            az_shear_llsd, div_shear=transform.llsd(myfinaldata, 3, 15, 1, 1.5, 0.5)
            az_shear, az_gate_shear = transform.az_cd(mfd_conv, nyquist, 0.8*nyquist, resolution, 2)
            az_shear[:,:41]=az_shear_llsd[:,:41]
            az_shear_conv=transform.conv(az_shear)
            p_mask=meso.mask(mask,coord, radar, cartesian, r, el); #p_mask[:]=1
            l_mask=meso.mask(newlabels,coord, radar, cartesian, r, el)
            # az_shear=az_shear*p_mask
            # az_shear_conv=az_shear_conv*p_mask
            # rotation_pos=variables.meso(); rotation_neg=variables.meso()
            ids=np.unique(l_mask)
            ids=ids[ids>0]
            # for ii in ids:
            #     binary=l_mask==ii
            #     binary_dil=skim.binary_dilation(binary, selem=np.ones([5,5]))*1
                
            #     az_shear_m=az_shear*binary_dil
            #     az_shear_conv_m=az_shear_conv*binary_dil
            #     myfinaldata_m=myfinaldata*binary_dil
            #     mfd_conv_m=mfd_conv*binary_dil
            #     if np.nanmax(abs(az_shear_m.flatten('C')))>=3 or ii==ids[0]:
            #         print("Identifying anticyclonic shears")
                    
            #         rotation_pos=meso.shear_group(rotation_pos, 1, 
            #                                                     mfd_conv_m, 
            #                                                     az_shear_m, 
            #                                                     ii, 
            #                                                     resolution, 
            #                                                     distance, 
            #                                                     shear, radar,
            #                                                     radar["elevations"][el], el,
            #                                                     radar["radars"][r], r,
            #                                                     cartesian["indices"], timelist[t])
                    
            #         rotation_neg=meso.shear_group(rotation_neg, -1, 
            #                                                     mfd_conv_m, 
            #                                                     az_shear_m, 
            #                                                     ii, 
            #                                                     resolution, 
            #                                                     distance, 
            #                                                     shear, radar,
            #                                                     radar["elevations"][el], el,
            #                                                     radar["radars"][r], r,
            #                                                     cartesian["indices"], timelist[t])

            if el<8:
                print('PLOTTING')
                # myfinaldata=myfinaldata*p_mask
                # mfd_conv=mfd_conv*p_mask
                # az_shear_conv=az_shear_conv*p_mask
                # myref=p_mask*myref
                # rot=np.nansum([rotation_pos["shear_objects"][el],-rotation_neg["shear_objects"][el]], axis=0)
                # rot[rot==0]=np.nan
                myfinaldata=myfinaldata[:,:160]
                mfd_conv=mfd_conv[:,:160]
                az_shear_conv=az_shear_conv[:,:160]
                myref=myref[:,:160]
                myref[np.isnan(myfinaldata)]=np.nan
                image=variables.imvars(nyquist, myfinaldata, resolution, specs,
                            str(timelist[t]), radar["elevations"][el],
                            radar["radars"][r])
    
                plot.plot_ppi_MF_masked(image["grid"][1], 
                                      image["grid"][0], myref, 
                                      0, 60, 
                                      turbo, np.arange(-1,62), 
                                      image["title"]+'ref', 
                                      path["images"], "ref"+image["name"])
                plot.plot_ppi_MF_masked(image["grid"][1], 
                                      image["grid"][0], mfd_conv, 
                                      -33, 33, 
                                      plt.cm.seismic, np.arange(-33,34), 
                                      image["title"]+"vel", 
                                      path["images"], "vel"+image["name"])
                plot.plot_ppi_MF_masked(image["grid"][1], 
                                      image["grid"][0], az_shear_conv, 
                                      image["vmin"], image["vmax"], 
                                      image["cmap"], image["bound"], 
                                      image["title"]+"az", 
                                      path["images"], "az"+image["name"])
                # plot.plot_ppi_MF_masked(image["grid"][1], 
                #                       image["grid"][0], rot, 
                #                       -1.5, 1.5, 
                #                       image["cmap"], np.arange(-2,3), 
                #                       image["title"]+"objects", 
                #                       path["images"], "ob"+image["name"])
                    
    vert_p, v_ID_p = meso.tower(rotation_pos, newlabels, radar, shear, r, timelist[t], path)
    vert_n, v_ID_n = meso.tower(rotation_neg, newlabels, radar, shear, r, timelist[t], path)
    ##%%
    towers_p=towers_p.append(vert_p)
    towers_n=towers_n.append(vert_n)
    
    tower_list_p.append(towers_p)
    tower_list_n.append(towers_n)
    r_toc=timeit.default_timer()
    print("time elapsed [s]: ", r_toc-r_tic)
    # contours=list(cells_id.cont)
    # cf_list=[]
    # for c in contours:
    #     for cont in c:
    #         cf_list.append(cont)
    # contours=list(cell_list[-2].cont)
    # c_list=[]
    # for c in contours:
    #     for cont in c:
    #         c_list.append(cont)
    # # plot.plot_cart_contour_quiver(ref_c[t,:,:], c_list, cf_list, flowfield, timelist[t], path["images"],
    # #                       str(event)+'_'+str(timelist[t])+'_flow.png', radar)
   
    # # print("Saving rotation towers to file: ", path["images"]+'towers_' + str(timelist[t]) + '.txt')
    # plot.plot_cart_scatter(ref_c[t,:,:], towers_p["x"],towers_p["y"], towers_p["size"].astype(float),
    #                     towers_n["x"],towers_n["y"], towers_n["size"].astype(float), 
    #                     cartesian["x"][0], cartesian["x"][-1], cartesian["y"][0], cartesian["y"][-1],
    #                     towers_p["r_max"], towers_n["r_max"], cf_list, timelist[t],
    #                     path["images"], str(event)+'_'+str(timelist[t])+'_cont.png', radar)
    # # plot.plot_cart_quiver(ref_c[t,:,:], flowfield, timelist[t], path["images"],
    # #                       str(event)+'_'+str(timelist[t])+'_quiv.png', radar)
#%%
IDs=np.unique(trt_cells)
p_track1=meso.summarise_rot(tower_list_p, IDs)
n_track1=meso.summarise_rot(tower_list_n, IDs)
#%%
# plot.plot_rottrack(p_track1, n_track1, 'test', path["images"], event+'rot.png', radar)
