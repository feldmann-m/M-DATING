#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:05:27 2020

@author: feldmann
"""
print('importing pyart readers')
from pyart.aux_io import read_file_py
from pyart.aux_io import read_metranet
from pyart.aux_io import read_cartesian_metranet
print('imported pyart readers')
import numpy as np
import os
import sys
from datetime import datetime
from datetime import timedelta
import pandas as pd
from zipfile import ZipFile
import shutil
from skimage.draw import polygon
import library.transform as transform
import geojson as gs
import geopandas as gpd
from geojson import FeatureCollection
import glob
#%%
def blockPrint():
    """
    subpresses print statements
    Returns
    -------
    None.

    """
    sys.stdout = open(os.devnull, 'w')
    
def enablePrint():
    """
    enables print statements
    Returns
    -------
    None.

    """
    sys.stdout = sys.__stdout__

def unzipvel(path,event):
    """
    unzip velocity data (DV-files)

    Parameters
    ----------
    path : dict
        contains data source and destination path.
    event : int
        year + day of year of event (YYDDD).

    Returns
    -------
    None.

    """
    for r in ['A','D','L','P','W']:
        dv=path["r2d2"]+'DV'+r+str(event)+'.zip'
        try:
            with ZipFile(dv, 'r') as zipObj:
                zipObj.extractall(path["temp"])
            dv=path["r2d2"]+'DV'+r+str(event-1)+'.zip'
            with ZipFile(dv, 'r') as zipObj:
                zipObj.extractall(path["temp"])
        except: print("directory "+ dv + " does not exist")
        try: shutil.move(path["temp"]+'srn/data/tmp/mof/DV'+r+'/', path["temp"])
        except: print("directory "+ path["temp"]+'srn/data/tmp/mof/DV'+r+'/' + " does not exist")
    
def unzipc(path,event):
    """
    unzip reflectivity files (CZ-files)

    Parameters
    ----------
    path : dict
        contains data source and destination path.
    event : int
        year + day of year of event (YYDDD).

    Returns
    -------
    None.

    """
    cz=path["archive"]+str(event)+'/CZC'+str(event)+'.zip'
    try:
        with ZipFile(cz, 'r') as zipObj:
            zipObj.extractall(path["temp"])
    except: print("directory "+ cz + " does not exist")
    
def unzip(path,event):
    """
    unzip reflectivity and velocity files (CZ, DV, and ML-files)

    Parameters
    ----------
    path : dict
        contains data source and destination path.
    event : int
        year + day of year of event (YYDDD).

    Returns
    -------
    None.

    """
    cz=path["archive"]+str(event)+'/CZC'+str(event)+'.zip'
    try:
        with ZipFile(cz, 'r') as zipObj:
            zipObj.extractall(path["temp"])
    except: print("directory "+ cz + " does not exist")
    cz_prev=path["archive"]+str(event-1)+'/CZC'+str(event-1)+'.zip'
    try:
        with ZipFile(cz_prev, 'r') as zipObj:
            filelist=zipObj.namelist()
            for file in filelist:
                if str(event-1)+'23' in file:
                    zipObj.extract(file,path["temp"])
    except: print("directory "+ cz_prev + " does not exist")
    cz_prox=path["archive"]+str(event+1)+'/CZC'+str(event+1)+'.zip'
    try:
        with ZipFile(cz_prox, 'r') as zipObj:
            filelist=zipObj.namelist()
            for file in filelist:
                if str(event+1)+'00' in file:
                    zipObj.extract(file,path["temp"])
    except: print("directory "+ cz_prox + " does not exist")
    for r in ['A','D','L','P','W']:
        ml=path["archive"]+str(event)+'/ML'+r+str(event)+'.zip'
        ml_prev=path["archive"]+str(event-1)+'/ML'+r+str(event-1)+'.zip'
        ml_prox=path["archive"]+str(event+1)+'/ML'+r+str(event+1)+'.zip'
        try:
            with ZipFile(ml, 'r') as zipObj:
                zipObj.extractall(path["temp"])
        except: print("directory "+ ml + " does not exist")
        try:
            with ZipFile(ml_prev, 'r') as zipObj:
                filelist=zipObj.namelist()
                for file in filelist:
                    if str(event-1)+'23' in file:
                        zipObj.extract(file,path["temp"])
        except: print("directory "+ ml_prev + " does not exist")
        try:
            with ZipFile(ml_prox, 'r') as zipObj:
                filelist=zipObj.namelist()
                for file in filelist:
                    if str(event+1)+'00' in file:
                        zipObj.extract(file,path["temp"])
        except: print("directory "+ ml_prox + " does not exist")
    for r in ['A','D','L','P','W']:
        dv=path["r2d2"]+'DV'+r+str(event)+'.zip'
        dv_prev=path["r2d2"]+'DV'+r+str(event-1)+'.zip'
        dv_prox=path["r2d2"]+'DV'+r+str(event+1)+'.zip'
        try:
            with ZipFile(dv, 'r') as zipObj:
                zipObj.extractall(path["temp"])
        except: print("directory "+ dv + " does not exist")
        try:
            with ZipFile(dv_prev, 'r') as zipObj:
                filelist=zipObj.namelist()
                for file in filelist:
                    if str(event-1)+'23' in file:
                        zipObj.extract(file,path["temp"])
        except: print("directory "+ dv_prev + " does not exist")
        try:
            with ZipFile(dv_prox, 'r') as zipObj:
                filelist=zipObj.namelist()
                for file in filelist:
                    if str(event+1)+'00' in file:
                        zipObj.extract(file,path["temp"])
        except: print("directory "+ dv_prox + " does not exist")
        try: shutil.move(path["temp"]+'srn/data/tmp/mof/DV'+r+'/', path["temp"])
        except: print("directory "+ path["temp"]+'srn/data/tmp/mof/DV'+r+'/' + " does not exist")

def unzip_archive(apath,datapath,event,year,ID):
    """
    unzips variable from radar archive

    Parameters
    ----------
    apath : string
        archive path.
    datapath : string
        destination path.
    event : string
        date in YYDDD format.
    year : string
        year in YYYY format.
    ID : string
        Variable ID in archive.

    Returns
    -------
    None.

    """
    folder=apath+str(event)+'/'+ID+str(event)+'.zip'
    try:
        with ZipFile(folder, 'r') as zipObj:
            zipObj.extractall(datapath)
    except: print("directory "+ folder + " does not exist")
        
def rmfiles(path):
    """
    removes directory

    Parameters
    ----------
    path : dict
        contains path to be removed.

    Returns
    -------
    None.

    """
    try: shutil.rmtree(path["temp"])
    except: print("directory "+ path["temp"]+ " does not exist")
        
def makedir(path):
    """
    make directories for images

    Parameters
    ----------
    path : dict
        path for images.

    Returns
    -------
    None.

    """
    try:
        os.mkdir(path["images"])#+path["event"])
        print('Directory created')
    except FileExistsError:
        print('Directory already exists')
    try:
        os.mkdir(path["files"])
        print('Directory created')
    except FileExistsError:
        print('Directory already exists')
    try:
        os.mkdir(path["temp"])
        print('Directory created')
    except FileExistsError:
        print('Directory already exists')

def timelist(dates):
    """
    make list of dates in 5 minute intervals during event

    Parameters
    ----------
    dates : list
        start and end date of processing.

    Returns
    -------
    timelist : list
        timesteps in 5 min increments during processing.

    """
    start, end = [datetime.strptime(_, "%y%j%H%M") for _ in dates]
    date=start
    timelist = []
    while date<end:
        date=date+timedelta(minutes=5)
        timelist.append(date.strftime("%y%j%H%M"))

    return timelist


def read_del_data(file):
    """
    Reads MCH dealiased data (DV-files)

    Parameters
    ----------
    file : string
        filename including path.

    Returns
    -------
    array or None
        data extracted by reader, None if failed.
    int
        flag, whether data was successfully extracted.

    """
    print('Reading dealiased data: ', file)
    try:
        myfile = read_file_py(file,physic_value = True)
        mydata = myfile.data
        myheader = myfile.header
        nyq=float(myheader['nyquist'])
        myfinaldata = transform_from_digital(mydata, nyq)
        return myfinaldata, 1
    except:
        print("Data cannot be opened", file)
        return None, -1

def read_raw_data(file):
    """
    Reads MCH polarimetric data (ML-files), returns velocity and reflectivity

    Parameters
    ----------
    file : string
        filename including path.

    Returns
    -------
    array or None
        velocity data extracted by reader, None if failed.
    array or None
        reflectivity data extracted by reader, None if failed.
    int
        flag, whether data was successfully extracted.

    """
    print('Reading METRANET data: ', file)
    try:
        myfile=read_metranet(file,reader='python')
        myvel=myfile.get_field(0,'velocity')
        myref=myfile.get_field(0,'reflectivity').data
    
        myvel=myvel.data
        
        return myvel, myref, 1
    except:
        print("Data cannot be opened", file)
        return None, None, -1
    
def read_cartesian(file):
    """
    Reads Cartesian max-echo composite file

    Parameters
    ----------
    file : string
        filename including path.

    Returns
    -------
    array or None
        data extracted by reader, None if failed.
    int
        flag, whether data was successfully extracted.

    """
    print('Reading cartesian data: ', file)
    try:
        myfile=read_cartesian_metranet(file, reader='C')
        mydata=myfile.fields
        myref=mydata["maximum_echo"]["data"].data[0,:,:]
        return myref, 1
    except:
        print("Data cannot be opened, trying different ending", file)
        file=file[:-6]+'U'+file[-5:]
        try:
            myfile=read_cartesian_metranet(file, reader='C')
            mydata=myfile.fields
            myref=mydata["maximum_echo"]["data"].data[0,:,:]
            return myref, 1
        except:
            print("Data cannot be opened", file)
            return None, -1

def transform_from_digital(mydata, nyquist):
    """
    transforms digital number to velocity

    Parameters
    ----------
    mydata : array
        velocity data as digital number.
    nyquist : float
        nyquist velocity / bounding velocity.

    Returns
    -------
    myfinaldata : array
        velocity data in m/s.

    """
    ##transform digital number to Doppler velocity (MCH)
    myfinaldata=np.zeros((mydata.shape))
    for n1 in range(0,mydata.shape[0]):
         for n2 in range(0,mydata.shape[1]):
             if mydata[n1,n2] == 0:
                 myfinaldata[n1,n2]=np.nan
             else:
                 myfinaldata[n1,n2]= (mydata[n1,n2]-128)*nyquist/127
    return myfinaldata



def read_track(file):
    """
    reads tabular data from text file

    Parameters
    ----------
    file : string
        filename including path.

    Returns
    -------
    data : pandas dataframe
        tabular data.

    """
    data=pd.read_csv(file, sep=' ')
    data= data.drop(columns='Unnamed: 0')
    return data

def write_track(track, file):
    """
    writes dataframe to text file

    Parameters
    ----------
    track : pandas dataframe
        tabular data.
    file : string
        filename including path.

    Returns
    -------
    None.

    """
    track.to_csv(file, header=track.columns, index=range(len(track)), sep=' ', mode='a')
    
def TRT_to_grid(year, event, path):
    """
    Extracts contours from TRT cells and produces gridded product for entire day

    Parameters
    ----------
    year : string
        year in YYYY.
    event : date in
        YYDDD.
    path : dict
        dict containing all paths.

    Returns
    -------
    cellist : list of arrays
        list of all 2D gridded TRT cells.
    timelist : list
        list of all valid timesteps.

    """
    o_x=254000
    o_y=-159000
    lx=710; ly=640
    
    unzip_archive(path['archive']+event,path['temp'],event,year,'TRTC')
    cpath='/store/mch/msrad/radar/swiss/data/'+year+'/'+event
    cellist=[]; timelist=[]
    for r, d, f in os.walk(path['temp']):
        f=sorted(f,key=str.lower)
        for file in f:
            cells=np.zeros([ly,lx])
            if 'TRT' in file and event in file:
                print(file)
                data=pd.read_csv(path['temp']+file).iloc[8:]
                for n in range(len(data)):
                    t=data.iloc[n].str.split(';',expand=True)
                    TRT_ID=int(t[0].values)
                    time=int(t[1].values)
                    tt=np.array(t)[0,27:-1]
                    tt=np.reshape(tt,[int(len(tt)/2),2])
                    tlat=tt[:,1].astype(float); tlon=tt[:,0].astype(float)
                    chx,chy=transform.c_transform(tlon,tlat)
                    ix=np.round((chx-o_x)/1000).astype(int)
                    iy=np.round((chy-o_y)/1000).astype(int)
                    rr, cc = polygon(iy, ix, cells.shape)
                    cells[rr,cc]=int(t[0].values);
            if np.nansum(cells.flatten())>0:cellist.append(cells); timelist.append(time)
    return cellist, timelist


def read_TRT(path, file=0, ttime=0):
    """
    Read .trt or .json file containing TRT output
    Returns dataframe with attributes and gridded TRT cells

    Parameters
    ----------

    path : string
        path, where to look for files.
    file: string
        filename
    ttime : string
        timestep to find files for.
    Requires either filename or timestep
    
    Returns
    -------
    trt_df : dataframe
        TRT cells and attributes of the timestep.
    cells: list
        Gridded TRT cells per timestep
    timelist: list
        timesteps

    """
    
    o_x=254000
    o_y=-159000
    lx=710; ly=640
    cells=np.zeros([ly,lx])
    if file == 0:
        file=glob.glob(path["lomdata"]+'TRTC/*'+ttime+'*json*')
        if len(file)>0: flag=1
        else:
            file=glob.glob(path["lomdata"]+'TRTC/*'+ttime+'*'+'.trt')[0]
            flag=0
    else:
        if 'json' in file: flag=1; ttime=file[-20:-11]
        else: flag=0; ttime=file[-15:-6]
        file=[file]
    
    if flag==1:
        with open(file[0]) as f: gj = FeatureCollection(gs.load(f))
        trt_df=gpd.GeoDataFrame.from_features(gj['features'])
        if len(trt_df)>0:
          # print(trt_df.lon.values.astype(float))
          chx, chy = transform.c_transform(trt_df.lon.values.astype(float),trt_df.lat.values.astype(float))
          trt_df['chx']=chx.astype(str); trt_df['chy']=chy.astype(str)
          for n in range(len(trt_df)):
              lon,lat=trt_df.iloc[n].geometry.boundary.xy
              # print(trt_df.iloc[n])
              chx, chy = transform.c_transform(lon,lat)
              # trt_df.iloc[n]['chx']=chx.astype(str); trt_df.iloc[n]['chy']=chy.astype(str)
              #transform.c_transform(trt_df.iloc[n].lon.values,trt_df.iloc[n].lat.values)
              ix=np.round((chx-o_x)/1000).astype(int)
              iy=np.round((chy-o_y)/1000).astype(int)
              rr, cc = polygon(iy, ix, cells.shape)
              # print(lat,lon,chx,chy,ix,iy)
              cells[rr,cc]=int(trt_df.traj_ID.iloc[n]);
        else: cells=[]
    else:
        data=pd.read_csv(file).iloc[8:]
        headers=pd.read_csv(file).iloc[7:8].iloc[0][0].split()
        trt_df=pd.DataFrame()
        for n in range(len(data)):
            t=data.iloc[n].str.split(';',expand=True)
            trt_df.loc[n,'traj_ID']=int(t[0].values)
            trt_df.loc[n,'time']=int(t[1].values)
            trt_df.loc[n,'lon']=t[2].values.astype(float)
            trt_df.loc[n,'lat']=t[3].values.astype(float)
            chx,chy=transform.c_transform([trt_df.loc[n,'lon']],[trt_df.loc[n,'lat']])
            ix=np.round((chx-o_x)/1000).astype(int)
            if ix>=710: ix=709
            iy=np.round((chy-o_y)/1000).astype(int)
            if iy>=640: iy=639
            n2=27
            if int(ttime)>=221520631: n2=82
            tt=np.array(t)[0,n2:-1]
            tt=np.reshape(tt,[int(len(tt)/2),2])
            trt_df.loc[n,'chx']=chx
            trt_df.loc[n,'chy']=chy
            lat=tt[:,1].astype(float); lon=tt[:,0].astype(float)
            # trt_df=trt_df.astype(str)
            chx,chy=transform.c_transform(lon,lat)
            ix=np.round((chx-o_x)/1000).astype(int)
            iy=np.round((chy-o_y)/1000).astype(int)
            rr, cc = polygon(iy, ix, cells.shape)
            cells[rr,cc]=int(t[0].values);
    # print(np.nanmax(cells))
    timelist=[str(ttime)]
    return trt_df, [cells], timelist


def get_TRT(ttime, path):
    """
    Extracts contours from TRT cells and produces gridded product

    Parameters
    ----------
    year : string
        year in YYYY.
    event : date in
        YYDDD.
    path : dict
        dict containing all paths.

    Returns
    -------
    cellist : list of arrays
        list of all 2D gridded TRT cells.
    timelist : list
        list of all valid timesteps.

    """
    o_x=254000
    o_y=-159000
    lx=710; ly=640
    # cells=np.zeros([ly,lx])
    #cpath='/store/mch/msrad/radar/swiss/data/'+year+'/'+event
    cellist=[]; timelist=[]
    cells=np.zeros([ly,lx])
    file=glob.glob(path["lomdata"]+'TRTC/*'+ttime+'*'+'.trt')[0]
    cells=np.zeros([ly,lx])
    print(file)
    data=pd.read_csv(file).iloc[8:]
    for n in range(len(data)):
        t=data.iloc[n].str.split(';',expand=True)
        TRT_ID=int(t[0].values)
        time=int(t[1].values)
        if int(ttime)>220000000: tt=np.array(t)[0,82:-1]
        else: tt=np.array(t)[0,25:-1]
        tt=np.reshape(tt,[int(len(tt)/2),2])
        tlat=tt[:,1].astype(float); tlon=tt[:,0].astype(float)
        chx,chy=transform.c_transform(tlon,tlat)
        ix=np.round((chx-o_x)/1000).astype(int)
        iy=np.round((chy-o_y)/1000).astype(int)
        rr, cc = polygon(iy, ix, cells.shape)
        cells[rr,cc]=int(t[0].values);
    if np.nansum(cells.flatten())>0:cellist.append(cells); timelist.append(ttime)
    return cellist, timelist


def write_histfile(phist,nhist,path):
    """
    Overwrites history file with new mesocyclone continuity information

    Parameters
    ----------
    phist : dataframe
        positive mesocyclone continuity information.
    nhist : dataframe
        negative mesocyclone continuity information.
    path : string
        filepath to save histfile.

    Returns
    -------
    None.

    """
    file=glob.glob(path["outdir"]+'ROT/'+'*hist*')
    try: 
        if len(file)>0:
            for f in file: os.remove(f)
    except: print("directory "+ str(file)+ " does not exist")
    
    phist.to_csv(path["outdir"]+'ROT/'+'phist.txt', header=phist.columns, index=range(len(phist)), sep=';', mode='a')
    nhist.to_csv(path["outdir"]+'ROT/'+'nhist.txt', header=nhist.columns, index=range(len(nhist)), sep=';', mode='a')
    
def read_histfile(path):
    """
    Reads history file with new mesocyclone continuity information

    Parameters
    ----------
    path : string
        filepath to save histfile.

    Returns
    -------
    phist : dataframe
        positive mesocyclone continuity information.
    nhist : dataframe
        negative mesocyclone continuity information.

    """
    try:
        phist=pd.read_csv(path["outdir"]+'ROT/'+'phist.txt', sep=';')
        phist= phist.drop(columns='Unnamed: 0')
    except:
        phist=pd.DataFrame(data=None, index=None, columns=["ID","cont","dis","latest"])
    try:
        nhist=pd.read_csv(path["outdir"]+'ROT/'+'nhist.txt', sep=';')
        nhist= nhist.drop(columns='Unnamed: 0')
    except:
        nhist=pd.DataFrame(data=None, index=None, columns=["ID","cont","dis","latest"])
    return phist,nhist

def df_to_geojson(df, properties, lat='x', lon='y'):
    """
    Creates geojson with point geometry from dataframe

    Parameters
    ----------
    df : dataframe
        mesocyclone dataframe.
    properties : list
        names of properties in features collection.
    lat : string, optional
        replaces latitude with desired coordinate label. The default is 'x'.
    lon : string, optional
        replaces longitude with desired coordinate label. The default is 'y'.

    Returns
    -------
    geojson : TYPE
        DESCRIPTION.

    """
    # create a new python dict to contain our geojson data, using geojson format
    geojson = {'type':'FeatureCollection', 'features':[]}

    # loop through each row in the dataframe and convert each row to geojson format
    for _, row in df.iterrows():
        # create a feature template to fill in
        feature = {'type':'Feature',
                   'properties':{},
                   'geometry':{'type':'Point',
                               'coordinates':[]}}

        # fill in the coordinates
        llon, llat = transform.transform_c([row[lat]],[row[lon]])
        #print(llon, llat)
        feature['geometry']['coordinates'] = [llon[0], llat[0]]

        # for each column, get the value and add it as a new feature property
        for prop in properties:
            feature['properties'][prop] = row[prop]
        
        # add this feature (aka, converted dataframe row) to the list of features inside our dict
        geojson['features'].append(feature)
    
    return geojson

def write_geojson(tower_list,file):
    """
    Write geojson output

    Parameters
    ----------
    tower_list : list
        list containing mesocyclone dataframes.
    file : string
        filename.

    Returns
    -------
    None.

    """
    prop=list(tower_list.columns)
    prop.remove('radar')
    data=df_to_geojson(tower_list,prop)
    with open(file, 'w') as f:
        gs.dump(data['features'], f)
