#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:04:19 2023

@author: mfeldman

Creates plots for radarlive from realtime production, includes history of past 2h
run after realtime_parallel.py
"""
#%% import settings
import argparse as ap
parser = ap.ArgumentParser()
parser.add_argument('--dvdir', type=str, required=False,default='/srn/data/zuerh450/')
parser.add_argument('--lomdir', type=str, required=False,default='/srn/data/')
parser.add_argument('--outdir', type=str, required=False,default='/scratch/lom/mof/realtime/')
parser.add_argument('--codedir', type=str, required=False,default='/scratch/lom/mof/code/ELDES_MESO/')
parser.add_argument('--time', type=str, required=True)
args = parser.parse_args()
#%% import external libraries
import sys
sys.path.append(args.codedir)
import os
os.environ['METRANETLIB_PATH'] = '/srn/las/idl/lib/radlib/'
import pandas as pd
import skimage.morphology as skim
pd.options.mode.chained_assignment = None
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore',category=AstropyWarning)
import glob
import numpy as np
import geojson as gs
import geopandas as gpd
from geojson import FeatureCollection
import pyart
#%% import functions
import library.variables as variables
import library.plot as plot
import library.io as io
#%% Main function
def main():
    #import variables
    time=args.time
    
    radar, cartesian, path, specs, files, shear, resolution=variables.vars(args.dvdir,args.lomdir,args.outdir,args.codedir)
    #find TRT and rotation files of given day
    trtfiles=np.array(sorted(glob.glob(path["lomdata"]+'TRTC/*.json')))
    trtfile=np.array(sorted(glob.glob(path["lomdata"]+'TRTC/*'+time+'*.json')))
    pfiles=np.array(sorted(glob.glob(path["outdir"]+'ROT/'+'PROT*.json')))
    pfile=np.array(sorted(glob.glob(path["outdir"]+'ROT/'+'PROT*'+time+'*.json')))
    nfiles=np.array(sorted(glob.glob(path["outdir"]+'ROT/'+'NROT*.json')))
    nfile=np.array(sorted(glob.glob(path["outdir"]+'ROT/'+'NROT*'+time+'*.json')))
    i=np.where(trtfiles==trtfile)[0][0].astype(int)+1
    ii=np.where(pfiles==pfile)[0][0].astype(int)+1
    iii=np.where(nfiles==nfile)[0][0].astype(int)+1
    if np.nanmin([i,ii,iii])<12:
        trtfiles=trtfiles[:i]
        pfiles=pfiles[:ii]
        nfiles=nfiles[:iii]
    else:
        trtfiles=trtfiles[i-12:i]
        pfiles=pfiles[ii-12:ii]
        nfiles=nfiles[iii-12:iii]
    # pfiles=glob.glob(path["outdir"]+'ROT/'+'PROT*'+day+'*.json')
    # pfiles=sorted(pfiles)
    # nfiles=glob.glob(path["outdir"]+'ROT/'+'NROT*'+day+'*.json')
    # nfiles=sorted(nfiles)
    
    #%%initialize empty dataframes to append
    trtcells=pd.DataFrame()
    vert_p=pd.DataFrame(columns=['geometry', 'ID', 'time', 'x', 'y', 'dz', 'A', 'D', 'L', 'P', 'W',
           'A_range', 'D_range', 'L_range', 'P_range', 'W_range', 'A_n', 'D_n',
           'L_n', 'P_n', 'W_n', 'A_el', 'D_el', 'L_el', 'P_el', 'W_el', 'size_sum',
           'size_mean', 'vol_sum', 'vol_mean', 'z_0', 'z_10', 'z_25', 'z_50',
           'z_75', 'z_90', 'z_100', 'z_IQR', 'z_mean', 'r_0', 'r_10', 'r_25',
           'r_50', 'r_75', 'r_90', 'r_100', 'r_IQR', 'r_mean', 'v_0', 'v_10',
           'v_25', 'v_50', 'v_75', 'v_90', 'v_100', 'v_IQR', 'v_mean', 'd_0',
           'd_10', 'd_25', 'd_50', 'd_75', 'd_90', 'd_100', 'd_IQR', 'd_mean',
           'rank_0', 'rank_10', 'rank_25', 'rank_50', 'rank_75', 'rank_90',
           'rank_100', 'rank_IQR', 'rank_mean', 'cont', 'dist', 'flag'])
    vert_n=pd.DataFrame(columns=['geometry', 'ID', 'time', 'x', 'y', 'dz', 'A', 'D', 'L', 'P', 'W',
           'A_range', 'D_range', 'L_range', 'P_range', 'W_range', 'A_n', 'D_n',
           'L_n', 'P_n', 'W_n', 'A_el', 'D_el', 'L_el', 'P_el', 'W_el', 'size_sum',
           'size_mean', 'vol_sum', 'vol_mean', 'z_0', 'z_10', 'z_25', 'z_50',
           'z_75', 'z_90', 'z_100', 'z_IQR', 'z_mean', 'r_0', 'r_10', 'r_25',
           'r_50', 'r_75', 'r_90', 'r_100', 'r_IQR', 'r_mean', 'v_0', 'v_10',
           'v_25', 'v_50', 'v_75', 'v_90', 'v_100', 'v_IQR', 'v_mean', 'd_0',
           'd_10', 'd_25', 'd_50', 'd_75', 'd_90', 'd_100', 'd_IQR', 'd_mean',
           'rank_0', 'rank_10', 'rank_25', 'rank_50', 'rank_75', 'rank_90',
           'rank_100', 'rank_IQR', 'rank_mean', 'cont', 'dist', 'flag'])
    #read TRT and rotation files of day and add to dataframes
    for file in trtfiles:
        print(file)
        tdat,tcells,timelist=io.read_TRT(path,file=file)
        trtcells=pd.concat((trtcells,tdat),axis=0)#trtcells.append(tdat)
    for nfile in nfiles:
        with open(nfile) as f: gj = FeatureCollection(gs.load(f))
        vert_n=pd.concat((vert_n,gpd.GeoDataFrame.from_features(gj['features'])),axis=0)#vert_n.append(gpd.GeoDataFrame.from_features(gj['features']))
        
    for pfile in pfiles:
        with open(pfile) as f: gj = FeatureCollection(gs.load(f))
        vert_p=pd.concat((vert_p,gpd.GeoDataFrame.from_features(gj['features'])),axis=0)#vert_p.append(gpd.GeoDataFrame.from_features(gj['features']))
    print(vert_p); print(vert_n); print(trtcells)
    #%%read TRT file of timestep, use to cut out precipitation data -> displayed in background
    cells,timelist=io.get_TRT(time, path)
    if len(cells)>0:
      try:
        b_file=glob.glob(path["lomdata"]+'CZC/*'+str(time)+'*')[0]
        print(b_file)
        metranet=pyart.aux_io.read_cartesian_metranet(b_file)
        czc=metranet.fields['maximum_echo']['data'][0,:,:]
        #newcells=skim.dilation(cells[0],footprint=np.ones([5,5]))
        #newcells[newcells==0]=np.nan
        #newcells[newcells>0]=1
        #background=newcells*czc
        import copy
        background=copy.deepcopy(czc)
        background[background<0]=np.nan
      except:
        b_file=glob.glob(path["lomdata"]+'CZC/*'+str(time)+'*')[0]
        print('Problem with file',b_file)
        newcells=skim.dilation(cells[0],footprint=np.ones([5,5]))
        newcells[newcells==0]=np.nan
        newcells[newcells>0]=1
        background=newcells
    #if generation of background fails, is generated empty
    else:
      background=np.zeros([640,710])
      background[:]=np.nan
    #%% generate plot
    imtitle='Detected mesocyclones on VIL background';savepath=path["outdir"]+'IM/';imname='ROT'+str(time+'.png')
    plot.plot_cart_hist(time,background,trtcells,vert_p,vert_n, imtitle, savepath, imname, radar)

#%% CALL MAIN FUNCTION

if __name__ == "__main__":
    main()