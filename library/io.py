#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:05:27 2020

@author: feldmann
"""

from pyart.aux_io import read_file_py
from pyart.aux_io import read_metranet
from pyart.aux_io import read_cartesian_metranet
import numpy as np
import os
import sys
from datetime import datetime
from datetime import timedelta
import pandas as pd
from zipfile import ZipFile
import shutil
from skimage.draw import polygon
import transform
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
