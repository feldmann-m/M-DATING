#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:23:03 2020

@author: mfeldman
"""

import multiprocessing
import numpy as np
import argparse as ap
import pandas as pd
#%%
# parser = ap.ArgumentParser()
# parser.add_argument('--dvdir', type=str, required=False,default='/srn/data/zuerh450/')
# parser.add_argument('--lomdir', type=str, required=False,default='/srn/data/')
# parser.add_argument('--outdir', type=str, required=False,default='/scratch/lom/mof/realtime/')
# parser.add_argument('--codedir', type=str, required=False,default='/scratch/lom/mof/code/ELDES_MESO/')
# parser.add_argument('--time', type=str, required=True)
# args = parser.parse_args()

#%%
args=pd.DataFrame()
args.dvdir='/scratch/mfeldman/realtime/'
args.lomdir='/scratch/mfeldman/realtime/'
args.outdir='/scratch/mfeldman/realtime/'
args.codedir='/users/mfeldman/scripts/ELDES_MESO/'
args.time='221790745'
#%%

import os
import sys
sys.path.append('/users/mfeldman/scripts/ELDES_MESO')
sys.path.append(args.codedir)
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
import pyart
#%%

#%% INITIALIZE PROCESSING
# load case dates and times, load variables, launch timer
time=args.time
#event=sys.argv[2]
#year=sys.argv[3]
radar, cartesian, path, specs, files, shear, resolution=variables.vars(args.dvdir,args.lomdir,args.outdir,args.codedir)
coord=variables.read_mask(radar)
#io.makedir(path)
try:
    os.mkdir(args.outdir+'/ROT/')
    print('Directory created')
except FileExistsError:
    print('Directory already exists')

try:
    os.mkdir(args.outdir+'/IM/')
    print('Directory created')
except FileExistsError:
    print('Directory already exists')

tower_list_p=[]
tower_list_n=[]
#%% PROCESSING CURRENT TIMESTEPS
# launch parallelized rotation detection

t=0
trt_cells, timelist= io.get_TRT(time,path)
t_tic=timeit.default_timer()
#doy=timelist[t][:5]
if len(trt_cells)>0:
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
  
  rotation_pos=variables.meso()
  rotation_neg=variables.meso()
  
  for r in radar["n_radars"][2:3]:
      print("Analysing radar: ", radar["radars"][r])
      # PARALLEL ELEVATION PROCESSING
      for el in radar["n_elevations"][6:7]-1:
          print("Analysing sweep: ", radar["elevations"][el])
          # READ VELOCITY DATA
          dvfile=glob.glob(path["dvdata"]+'DV'+radar["radars"][r]+'/*'+timelist[t]+'*.8'+radar["elevations"][el])[0]
          # dvfile=path["temp"]+'DV'+radar["radars"][r]+'/DV'+radar["radars"][r]+timelist[t] \
          #         +'7L'+specs["sweep_ID_DV"]+radar["elevations"][el]
          myfinaldata, flag1 = io.read_del_data(dvfile)
          #COMPUTE MASK FROM TRT CONTOURS
          p_mask=meso.mask(mask,coord, radar, cartesian, r, el)
          l_mask=meso.mask(newlabels, coord, radar, cartesian, r, el)
          # exit if too few valid pixels or no velocity data
          if np.nansum(p_mask.flatten())<6: continue
          elif flag1 == -1: continue
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

      
  # MERGE OBJECT DETECTION FROM RADARS
  vert_p, v_ID_p = meso.tower(rotation_pos, newlabels, radar, shear, r, timelist[t], path)
  vert_n, v_ID_n = meso.tower(rotation_neg, newlabels, radar, shear, r, timelist[t], path)
  
  r_toc=timeit.default_timer()
  print("time elapsed [s]: ", r_toc-r_tic)
  
else:
  vert_p=pd.DataFrame(columns=["ID", "time", "radar","x", "y", "dz",
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
  vert_n=pd.DataFrame(columns=["ID", "time", "radar","x", "y", "dz",
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

tower_list_p.append(vert_p)
tower_list_n.append(vert_n)

t_toc=timeit.default_timer()
print("Computation time timestep: [s] ",t_toc-t_tic)
phist,nhist=io.read_histfile(path)
phist,vert_p=meso.rot_hist(vert_p, phist,time)
nhist,vert_n=meso.rot_hist(vert_n, nhist,time)
io.write_histfile(phist,nhist,path)
pfile=path["outdir"]+'ROT/'+'PROT'+str(time)+'.json'
io.write_geojson(vert_p,pfile)
nfile=path["outdir"]+'ROT/'+'NROT'+str(time)+'.json'
io.write_geojson(vert_n,nfile)
