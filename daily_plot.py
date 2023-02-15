#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:04:19 2023

@author: mfeldman
"""
import argparse as ap
#%%
parser = ap.ArgumentParser()
parser.add_argument('--dvdir', type=str, required=False,default='/srn/data/zuerh450/')
parser.add_argument('--lomdir', type=str, required=False,default='/srn/data/')
parser.add_argument('--outdir', type=str, required=False,default='/scratch/lom/mof/realtime/')
parser.add_argument('--codedir', type=str, required=False,default='/scratch/lom/mof/code/ELDES_MESO/')
parser.add_argument('--day', type=str, required=True)
args = parser.parse_args()
#%%
import numpy as np
import sys
sys.path.append('/users/mfeldman/scripts/ELDES_MESO')
sys.path.append('/scratch/lom/mof/code/ELDES_MESO')
import os
os.environ['METRANETLIB_PATH'] = '/srn/las/idl/lib/radlib/'
import pandas as pd
import skimage.morphology as skim
pd.options.mode.chained_assignment = None
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore',category=AstropyWarning)
import library.variables as variables
import library.plot as plot
import library.io as io
import glob
import pyart
import geojson as gs
import geopandas as gpd
from geojson import FeatureCollection
#%%
day=args.day

radar, cartesian, path, specs, files, shear, resolution=variables.vars(args.dvdir,args.lomdir,args.outdir,args.codedir)
trtfiles=glob.glob(path["lomdata"]+'TRTC/*'+day+'*.trt')[0]
trtfiles=sorted(trtfiles)
pfiles=glob.glob(path["outdir"]+'ROT/'+'PROT*'+day+'*.json')
pfiles=sorted(pfiles)
nfiles=glob.glob(path["outdir"]+'ROT/'+'NROT*'+day+'*.json')
nfiles=sorted(nfiles)

#%%
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
for file in trtfiles:
    tcells,timelist=io.get_TRT_file(file)
    trtcells=trtcells.append(tcells)
for nfile in nfiles:
    with open(nfile) as f: gj = FeatureCollection(gs.load(f))
    vert_n=vert_n.append(gpd.GeoDataFrame.from_features(gj['features']))
for pfile in pfiles:
    with open(pfile) as f: gj = FeatureCollection(gs.load(f))
    vert_p=vert_p.append(gpd.GeoDataFrame.from_features(gj['features']))
#%%


imtitle='Detected mesocyclones on VIL background';savepath=path["outdir"]+'IM/'; imname='DAYROT'+str(day)+'.png'
print(len(xp),len(xn))
plot.plot_cart_day(trtcells,vert_p,vert_n, imtitle, savepath, imname, radar)