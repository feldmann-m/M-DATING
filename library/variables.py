#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:55:58 2020

@author: feldmann
"""

import numpy as np
import numpy.matlib as npm
from netCDF4 import Dataset
import pandas as pd
# import pyart

#%%
def vars(dvdir,lomdir,outdir,codedir):#event, year):
    #RADAR PARAMETERS
    radar =	{
        "radars": ['A','D','L','P','W'],
        "n_radars": np.arange(0,5,1),
        "elevations": ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20'],
        "n_elevations": np.array(['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']).astype(int),
        "angles": np.array([-0.2, 0.4, 1.0, 1.6, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 11.0, 13.0, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0]),
        "nyquist": np.array([8.3,9.6,8.3,12.4,11.,12.4,13.8,12.4,13.8,16.5,16.5,16.5,20.6,20.6,20.6,20.6,20.6,20.6,20.6,20.6]),
        "x": [681201,497057,707957,603687,779700],
        "y": [237604,142408,99762,135476,189790],
        "z": [983,1682,1626,2937,2850],
        "elevation_ranges": np.asarray([246,210,246,162,183,162,146,162,140,121,111,100,87,75,62,50,41,34,30,27])
    }
    
    # CARTESIAN PARAMETERS
    cartesian = {
        # "grid": np.zeros([200,609,537]),
        "x": np.arange(297000,906000,1000),
        "y": np.arange(-100000,437000,1000),
        "z": np.arange(0,20000,100),
        "indices": [],
        "radar_ID": ['index_A','index_D','index_L','index_P','index_W'],
        "ox": 255000,
        "oy": -160000,
        "rx": np.round((np.array(radar["x"])-255000)/1000).astype(int),
        "ry": np.round((np.array(radar["y"])+160000)/1000).astype(int)
        
    }
    # cartesian["grid"][:]=np.nan
    # fg_ind=Dataset('/scratch/mfeldman/radar_indices/radar_indices_cartesian_regularz.nc', 'r')
    # for n in radar["n_radars"]:
    #     print(cartesian["radar_ID"][n])
    #     cartesian["indices"].append( fg_ind['radars'][cartesian["radar_ID"][n]][:].data )
        
    # PATH PARAMETERS
    path = {
        "home": '/users/mfeldman/',
        "scripts": codedir,#'/scratch/lom/mof/code/ELDES_MESO/',
        "dvdata": dvdir,#'/srn/data/zuerh450/',
        "lomdata": lomdir,#'/srn/data/',
        "outdir": outdir,#'/scratch/lom/mof/realtime/',
        #"mldata": '/scratch/mfeldman/mesocyclone_detection/c_cases/',
        #"czdata": '/scratch/mfeldman/mesocyclone_detection/c_cases/',
        #"images": '/store/mch/msrad/mfeldman/im_med/',
        #"files": '/store/mch/msrad/mfeldman/file_med/'+event+'/',
        #"event": event,
        #"archive": '/store/msrad/radar/swiss/data/'+year+'/',
        #"temp": '/scratch/mfeldman/temp_rot/'+event+'/',
        #"temp": '/scratch/mfeldman/realtime/',
        #"temp": '/scratch/lom/mof/realtime/',
        #"r2d2": '/store/mch/msrad/mfeldman/'+year+'/'
    }
    
    specs = {
        "sweep_ID_DV": '.8',
        "sweep_ID_ML": '.0',
        "test_ID": 'ex_',
        "resolution": 'low',
    }
    
    files = {
        "ML_files": [],
        "DV_files": []
    }
    
    #SHEAR PARAMETERS
    shear = {
        "near1": 3,
        "far1": 1,
        "near2": 6,
        "near2d": 3,
        "length": 1,
        "width": 1,
        "vort1": 0.01,
        "vort1d": 0.004,
        "rvel1": 10,
        "rvel1d": 4,
        "vortu": 0.01,
        "vortd": 0.004,
        "rotu": 10,
        "rotd": 4,
        "zu": 3000,
        "zd": 1000
        
    }
    if specs["resolution"]=='low': resolution=0.5;
    else: resolution=0.083;
    return radar, cartesian, path, specs, files, shear, resolution

def meso():
    obj = {
            "shear_objects": [],
            "prop": pd.DataFrame(columns=["ID", "time", "elevation", "radar", "indices", "x", "y", "z", "dvel", "vort", "diam", "rank", "v_ID", "size", "vol", "range"]),
            "shear_grid": np.zeros([360,512,20]),
            "shear_ID": [],
            "pattern_vector": []
            }
    return obj

def rot_df():
    df=pd.DataFrame(columns=["ID", "time", "radar","x", "y", "dz",
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
    return df

def distance(myfinaldata, resolution):
    distance=np.arange(0.5*resolution, myfinaldata.shape[1]*resolution 
                       + 0.5*resolution, resolution)
    distance=np.divide(np.multiply(distance,2*np.pi),360)
    distance=npm.repmat(distance,myfinaldata.shape[0],1)
    return distance

def time(file):
    dtime=file[-15:-6]
    yr=dtime[0:2]; dy=dtime[2:5]; hr=dtime[5:7]; mn=dtime[7:9]
    tstamp = '_' + str(yr) + '_' + str(dy) + '_' + str(hr) + str(mn); #timestamp as string
    time={
        "datetime": dtime,
        "tstamp": tstamp,
        "year": yr,
        "day": dy,
        "hour": hr,
        "min": mn
    }
    return time
#%%
def mask_coord(radar):
    import pyart
    azimuths=np.arange(0,360,1)
    coord=[]
    for e in range(20):
        el=radar['angles'][e]
        print(el)
        ranges=np.arange(0.25,radar["elevation_ranges"][e],0.5)
        c=np.zeros([3,len(azimuths),len(ranges)])
        for r in range(len(ranges)):
            for az in azimuths:
                x,y,z=pyart.core.antenna_to_cartesian(ranges[r], az, el)
                c[0,az,r]=x
                c[1,az,r]=y
                c[2,az,r]=z
        coord.append(c)
        np.save('mask_data/mask3d'+str(el)+'.npy',c)
    return coord

def read_mask(radar):
    azimuths=np.arange(0,360,1)
    coord=[]
    for e in range(20):
        el=radar['angles'][e]
        c=np.load('mask_data/mask3d'+str(el)+'.npy')
        coord.append(c)
    return coord
