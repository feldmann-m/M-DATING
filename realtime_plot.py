#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 16:24:04 2022

@author: mfeldman
"""
import argparse as ap
#%%
parser = ap.ArgumentParser()
parser.add_argument('--dvdir', type=str, required=False,default='/srn/data/zuerh450/')
parser.add_argument('--lomdir', type=str, required=False,default='/srn/data/')
parser.add_argument('--outdir', type=str, required=False,default='/scratch/lom/mof/realtime/')
parser.add_argument('--codedir', type=str, required=False,default='/scratch/lom/mof/code/ELDES_MESO/')
parser.add_argument('--time', type=str, required=True)
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

time=args.time
#event=sys.argv[2]
#year=sys.argv[3]
radar, cartesian, path, specs, files, shear, resolution=variables.vars(args.dvdir,args.lomdir,args.outdir,args.codedir)

#%%
pfile=path["outdir"]+'ROT/'+'PROT'+str(time)+'.json'
with open(pfile) as f: gj = FeatureCollection(gs.load(f))
vert_p=gpd.GeoDataFrame.from_features(gj['features'])
if len(vert_p)==0: vert_p=pd.DataFrame(columns=['geometry', 'ID', 'time', 'x', 'y', 'dz', 'A', 'D', 'L', 'P', 'W',
       'A_range', 'D_range', 'L_range', 'P_range', 'W_range', 'A_n', 'D_n',
       'L_n', 'P_n', 'W_n', 'A_el', 'D_el', 'L_el', 'P_el', 'W_el', 'size_sum',
       'size_mean', 'vol_sum', 'vol_mean', 'z_0', 'z_10', 'z_25', 'z_50',
       'z_75', 'z_90', 'z_100', 'z_IQR', 'z_mean', 'r_0', 'r_10', 'r_25',
       'r_50', 'r_75', 'r_90', 'r_100', 'r_IQR', 'r_mean', 'v_0', 'v_10',
       'v_25', 'v_50', 'v_75', 'v_90', 'v_100', 'v_IQR', 'v_mean', 'd_0',
       'd_10', 'd_25', 'd_50', 'd_75', 'd_90', 'd_100', 'd_IQR', 'd_mean',
       'rank_0', 'rank_10', 'rank_25', 'rank_50', 'rank_75', 'rank_90',
       'rank_100', 'rank_IQR', 'rank_mean', 'cont', 'dist', 'flag'])
nfile=path["outdir"]+'ROT/'+'NROT'+str(time)+'.json'
with open(nfile) as f: gj = FeatureCollection(gs.load(f))
vert_n=gpd.GeoDataFrame.from_features(gj['features'])
if len(vert_n)==0: vert_n=pd.DataFrame(columns=['geometry', 'ID', 'time', 'x', 'y', 'dz', 'A', 'D', 'L', 'P', 'W',
       'A_range', 'D_range', 'L_range', 'P_range', 'W_range', 'A_n', 'D_n',
       'L_n', 'P_n', 'W_n', 'A_el', 'D_el', 'L_el', 'P_el', 'W_el', 'size_sum',
       'size_mean', 'vol_sum', 'vol_mean', 'z_0', 'z_10', 'z_25', 'z_50',
       'z_75', 'z_90', 'z_100', 'z_IQR', 'z_mean', 'r_0', 'r_10', 'r_25',
       'r_50', 'r_75', 'r_90', 'r_100', 'r_IQR', 'r_mean', 'v_0', 'v_10',
       'v_25', 'v_50', 'v_75', 'v_90', 'v_100', 'v_IQR', 'v_mean', 'd_0',
       'd_10', 'd_25', 'd_50', 'd_75', 'd_90', 'd_100', 'd_IQR', 'd_mean',
       'rank_0', 'rank_10', 'rank_25', 'rank_50', 'rank_75', 'rank_90',
       'rank_100', 'rank_IQR', 'rank_mean', 'cont', 'dist', 'flag'])

#%%

#b_file=glob.glob(path["lomdata"]+'LZC/*'+str(time)+'*')
cells,timelist=io.get_TRT(time, path)
#print(np.unique(czc))
if len(cells)>0:
  try:
    b_file=glob.glob(path["lomdata"]+'RZC/*'+str(time)+'*')[0]
    print(b_file)
    metranet=pyart.aux_io.read_cartesian_metranet(b_file)
    czc=metranet.fields['radar_estimated_rain_rate']['data'][0,:,:]
    newcells=skim.dilation(cells[0],footprint=np.ones([5,5]))
    newcells[newcells==0]=np.nan
    newcells[newcells>0]=1
    background=newcells*czc
  except:
    b_file=glob.glob(path["lomdata"]+'RZC/*'+str(time)+'*')[0]
    print('Problem with file',b_file)
    newcells=skim.dilation(cells[0],footprint=np.ones([5,5]))
    newcells[newcells==0]=np.nan
    newcells[newcells>0]=1
    background=newcells

#  metranet=pyart.aux_io.read_cartesian_metranet(b_file,reader='python')
#  czc=metranet.fields['radar_estimated_rain_rate']['data'][0,:,:]
#  newcells=skim.dilation(cells[0],footprint=np.ones([5,5]))
#  newcells[newcells==0]=np.nan
#  newcells[newcells>0]=1
#  background=newcells*czc
  #background=metranet.fields['vertically_integrated_liquid']['data'][0,:,:]
else:
  background=np.zeros([640,710])
  background[:]=np.nan
xp=vert_p.x; yp=vert_p.y; sp=np.nansum([vert_p.A_n,vert_p.D_n,vert_p.L_n,vert_p.P_n,vert_p.W_n]);fp=vert_p.flag;cp=vert_p.rank_90
xn=vert_n.x; yn=vert_n.y; sn=np.nansum([vert_n.A_n,vert_n.D_n,vert_n.L_n,vert_n.P_n,vert_n.W_n]);fn=vert_n.flag;cn=vert_n.rank_90
imtitle='Detected mesocyclones on VIL background';savepath=path["outdir"]+'IM/';imname='ROT'+str(time+'.png')
print(len(xp),len(xn))
plot.plot_cart_obj(background, xp, yp, sp*20, fp, xn, yn, sn*20, fn, cp, cn, imtitle, savepath, imname, radar)
