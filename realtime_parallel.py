#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:49:37 2020

@author: mfeldman

Main script to process a single timestep

parallelized operations, not suited to run in an interactive editor

Called for realtime processing
"""

#%% INPUT OPTIONS
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--dvdir', type=str, required=False,default='/srn/data/zuerh450/')
parser.add_argument('--lomdir', type=str, required=False,default='/srn/data/')
parser.add_argument('--outdir', type=str, required=False,default='/scratch/lom/mof/realtime/')
parser.add_argument('--codedir', type=str, required=False,default='/scratch/lom/mof/code/ELDES_MESO/')
parser.add_argument('--time', type=str, required=True)
args = parser.parse_args()
#%% IMPORT LIBRARIES
import multiprocessing
import numpy as np
import os
import sys
sys.path.append('/users/mfeldman/scripts/ELDES_MESO')
sys.path.append(args.codedir)
import pandas as pd
pd.options.mode.chained_assignment = None
import skimage.morphology as skim
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore',category=AstropyWarning)
import timeit
import glob
#%% IMPORT FUNCTIONS
import library.variables as variables
import library.io as io
import library.transform as transform
import library.meso as meso

#%% MAIN FUNCTIONS
def main():
    # INITIALIZE PROCESSING
    # load case dates and times, load variables, launch timer
    time=args.time
    radar, cartesian, path, specs, files, shear, resolution=variables.vars(args.dvdir,args.lomdir,args.outdir,args.codedir)
    coord=variables.read_mask(radar)
    
    # make output directories
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

    # get thunderstorm data
    t=0
    trt_df, trt_cells, timelist= io.read_TRT(path,0,time)
    t_tic=timeit.default_timer()
    if len(trt_df)>0:
      # if any thunderstorm cells exist, continue algorithm
      newlabels=skim.dilation(trt_cells[0],footprint=np.ones([5,5]))
      
      
      print("starting rotation detection, analysing timestep: ", timelist[t])
      # ROTATION TRACKING
      r_tic=timeit.default_timer()
      
      rotation_pos=variables.meso(); rotation_neg=variables.meso()
      # Make unique radar-elevation IDs
      els=np.arange(1,21)
      rads=np.arange(100,501,100)
      rel=[]
      for r in rads:
          for el in els:
              rel.append(r+el)
      
      # Prepare parallel processes
      if __name__ == '__main__':
          manager = multiprocessing.Manager()
          return_dict = manager.dict()
          jobs = []
      
      for rel_i in rel:
          #Check if any thunderstorm in radar-elevation domain
          r=int(rel_i/100)-1; el=rel_i%100-1
          l_mask=meso.mask(newlabels,coord, radar, cartesian, r, el)
          
          if np.nansum((l_mask>0).flatten())>6:
              #Call parallel process, block print statements in parallelized section
              print('Launching process for ',r+1,el+1)
              io.blockPrint()
              p = multiprocessing.Process(target=radel_processor, args=(rel_i, radar, cartesian,
                                        path, specs, files, shear, resolution, timelist,
                                        t, l_mask, coord[el], return_dict))
              jobs.append(p)
              p.start()
              io.enablePrint()

      for proc in jobs:
          proc.join()
      # Join results from parallel processes
      result=return_dict.values()
      
      for n in range(0,len(result)):
          # Sort results into corresponding variables
          s_p, s_n = result[n]
          rotation_pos["shear_objects"].append(s_p["shear_objects"])
          rotation_pos["prop"]=pd.concat([rotation_pos["prop"],s_p["prop"]], ignore_index=True)
          rotation_pos["shear_ID"].append(s_p["shear_ID"])
          rotation_neg["shear_objects"].append(s_n["shear_objects"])
          rotation_neg["prop"]=pd.concat([rotation_neg["prop"],s_n["prop"]], ignore_index=True)
          rotation_neg["shear_ID"].append(s_n["shear_ID"])
          
      # MERGE OBJECT DETECTION FROM ALL RADARS AND ELEVATIONS -> vertical continuity check
      vert_p, v_ID_p = meso.tower(rotation_pos, newlabels, radar, shear, r, timelist[t], path)
      vert_n, v_ID_n = meso.tower(rotation_neg, newlabels, radar, shear, r, timelist[t], path)
      
      r_toc=timeit.default_timer()
      print("time elapsed rotation detection [s]: ", r_toc-r_tic)
      
    else:
      # In case of no thunderstorms, return empty result dataframe
      vert_p=variables.rot_df()
      vert_n=variables.rot_df()

    t_toc=timeit.default_timer()
    print("Computation time timestep: [s] ",t_toc-t_tic)
    # Read file with Mesocyclone history
    phist,nhist=io.read_histfile(path)
    # Time continuity and range check for new history file
    phist,vert_p=meso.rot_hist(vert_p, phist,time)
    nhist,vert_n=meso.rot_hist(vert_n, nhist,time)
    # Overwrite history file
    io.write_histfile(phist,nhist,path)
    # Write mesocyclone products
    pfile=path["outdir"]+'ROT/'+'PROT'+str(time)+'.json'
    io.write_geojson(vert_p,pfile)
    nfile=path["outdir"]+'ROT/'+'NROT'+str(time)+'.json'
    io.write_geojson(vert_n,nfile)



def radel_processor (rel, radar, cartesian, path, specs, files, shear, resolution,
                    timelist, t, l_mask, coord, return_dict):
    """
    parallel processing of radars and elevations

    Parameters
    ----------
    rel : int
        unique radar-elevation number.
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
    
    print('Radar-elevation is', rel)
    # Regain radar and elevation numbers
    r=int(rel/100)-1
    el=rel%100-1
    print("Analysing radar: ",r+1,", elevation: ",el+1)
    
    # Initialize rotation dictionaries
    rotation_pos=variables.meso()
    rotation_neg=variables.meso()
    
    # Find and read velocity file
    dvfile=glob.glob(path["dvdata"]+'DV'+radar["radars"][r]+'/*'+timelist[t]+'*.8'+radar["elevations"][el])[0]
    dvdata, flag1 = io.read_del_data(dvfile)
    
    # exit if no velocity data
    if flag1 == -1: 
        return variables.meso(), variables.meso();
    else:
        # derive azimuthal shear
        nyquist=radar["nyquist"][el]
        mfd_conv=transform.conv(dvdata)
        distance=variables.distance(dvdata, resolution)
        mfd_conv[:,40:]=dvdata[:,40:]
        az_shear = transform.az_cd(mfd_conv, nyquist, 0.8*nyquist, resolution, 2)[0]
        # initialize rotation variables
        rotation_pos=variables.meso(); rotation_neg=variables.meso()
        # get thunderstorm IDs
        ids=np.unique(l_mask)
        ids=ids[ids>0]
        # process each thunderstorm individually
        cellvar=l_mask, az_shear, mfd_conv, rotation_pos, rotation_neg, distance, resolution, shear, radar, coord, timelist, r, el
        for ii in ids:
            rotation_pos, rotation_neg = meso.cell_loop(ii, cellvar)

    #return results from parallel processing
    return_dict[r*20+el]= rotation_pos, rotation_neg



#%% CALL MAIN FUNCTION

if __name__ == "__main__":
    main()



