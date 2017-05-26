#!/usr/bin/env python
"""
Functions for downloading/processing verification datasets.

Available datasets with this module:
- GFS analyses (0.5 deg, 3-hourly)
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from ftplib import FTP
import os
from color_maker.color_maker import color_map

#############################################################################################################
### GENERAL TOOLS ###########################################################################################
#############################################################################################################

def convert_grb2nc(ncdir, nctable='cfs.table', outfile='cfsr.nc', daterange=None,
                   isanalysis=True, interp=True, gribtag=None):
    """
    Converts downloaded NCEP gribs (forecasts or analyses) to netcdf, 
    retaining only the dates, forecast types, and variables designated 
    in nctable file and the -match keyword
    
    Optionally, the gribs are all interpolated to a 0.5-degree lat-lon grid
    before conversion to netCDF (to ensure compatibility for combination)
    
    Requires the wgrib2 utility.
    """
    from subprocess import check_output, Popen
    import time
    from mpasoutput import timedelta_hours
    
    ncoutfile = '{}/{}'.format(ncdir, outfile)
    if os.path.isfile(ncoutfile):
        print('{} already exists!'.format(outfile))
        return
    # Point to the nc_table
    tablefile = '{}/{}'.format(ncdir, nctable)
    assert os.path.isfile(tablefile)
    
    # List the gribs to be converted
    if isanalysis and gribtag is None:
        gribtag = 'anl'
    elif gribtag is None:
        gribtag = 'fcst'
    grbfiles = check_output(['ls -1a {}/{}*.grb2'.format(ncdir, gribtag)], shell=True).split()
    grbfiles = [g.decode("utf-8") for g in grbfiles]
    
    # Use the -match keyword to select the desired dates
    with open(tablefile, "r") as tfile:
        for l, line in enumerate(tfile):
            if l==14:
                matchtag = line[1:].rstrip()
                break
    if isanalysis:
        # Match tags for CFSR analyses
        matchtag += ' -match ":(anl|0-6 hour ave fcst):"'
        # Add another -match tag for the desired date range
        if daterange is not None:
            idate = daterange[0]; fdate = daterange[1]
            currdate = idate
            matchtag2 = '-match ":('
            while currdate <= fdate:
                matchtag2 += 'd={:%Y%m%d%H}'.format(currdate)
                currdate += timedelta(hours=6)
                if currdate <= fdate: matchtag2 += '|'
            matchtag2 += '):"'
            matchtag = '{} {}'.format(matchtag, matchtag2)      
            
    else:
        assert daterange is not None
        # Match tags for CFSv2 forecasts (should include the init analysis)
        matchtag += ' -match ":d={:%Y%m%d%H}:"'.format(daterange[0])
        # Add another -match tag for the desired date range
        dhours = timedelta_hours(daterange[0], daterange[1])
        matchtag2 = '-match ":('
        for h in range(0, dhours+1, 6):
            matchtag2 += '{} hour fcst'.format(h)
            if h < dhours: matchtag2 += '|'
        matchtag2 += '):"'
        matchtag = '{} {}'.format(matchtag, matchtag2)      
                
    # Interpolate each grib file to a 0.5-deg lat-lon grid and then convert/append
    # to one netcdf file
    start = time.time()
    for g, grbfile in enumerate(grbfiles):
        print('processing grib {} of {}...'.format(g+1, len(grbfiles)))
        print(grbfile)
        if interp:
            print('  interpolating...')
            interpcomm = 'wgrib2 {} {} -new_grid latlon 0:720:0.5 -90:361:0.5 temp.grb2'
            interpcomm = interpcomm.format(grbfile, matchtag)
            Popen([interpcomm], shell=True).wait()
            convertcomm = 'wgrib2 temp.grb2 -nc_table {} -netcdf {}'
            convertcomm = convertcomm.format(tablefile, ncoutfile)
        else:
            convertcomm = 'wgrib2 {} {} -nc_table {} -netcdf {}'
            convertcomm = convertcomm.format(grbfile, matchtag, tablefile, ncoutfile)
        print('  converting to netcdf...')
        if os.path.isfile(ncoutfile):
            splits = convertcomm.split('-nc_table')
            convertcomm = splits[0] + '-append -nc_table' + splits[1]
        Popen([convertcomm], shell=True).wait()
        if interp: Popen(['rm -f temp.grb2'], shell=True).wait()
    end = time.time()
    print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    return

#############################################################################################################
### SECTION FOR HANDLING OPERATIONAL/RETROSPECTIVE CFSv2 FORECASTS/ANALYSES #################################
#############################################################################################################
cfs_vars = ['chi200', 'prate', 'pressfc', 'psi200', 'pwat', 'tmp2m',
            'tmpsfc', 'ulwtoa', 'wnd200', 'wnd850', 'z500', 'z200']

def download_cfsrr(idate, fdate, cfsdir, verbose=False, anlonly=False):
    """
    Downloads CFSv2 reforecasts (initalized on idate) and 
    corresponding analyses (CFSR; from idate to fdate)
    
    """
    from urllib.request import urlretrieve
    import time
    
    if not os.path.isdir(cfsdir):
        os.system('mkdir {}'.format(cfsdir))
    
    #==== First download the reanalyses ========================================
    print('\n==== Downloading CFSR reanalyses ====')
    nomads = 'https://nomads.ncdc.noaa.gov/data/cfsr'
    start = time.time()
    
    # create a list of tuples specifying the years and months encompassed
    # in the desired date range
    dts = list(set([(dt.year, dt.month) for dt in [idate+timedelta(days=d) for d in \
                                                   range((fdate-idate).days + 1)]]))
    for y,m in dts:
        for var in cfs_vars:
            # Download the forecast of each desired variable
            yyyymm = '{:04d}{:02d}'.format(y, m)
            url = '{}/{}/{}.gdas.{}.grb2'.format(nomads, yyyymm, var, yyyymm)
            localfile = '{}/anl.{}.cfs.{:02d}.grb2'.format(cfsdir, var, m)
            if os.path.isfile(localfile):
                if verbose: print('File already exists:\n{}'.format(localfile))
                continue
            try:
                # Download the forecast
                if verbose: print('Downloading {}...'.format(url))
                urlretrieve(url, localfile)
            except:
                if verbose: print('{} not found'.format(var))
    end = time.time()
    print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    
    # if we only want the reanalyses, then we're done!
    if anlonly: return
    
    #==== Now download the reforecasts ========================================
    print('\n==== Downloading CFSv2 reforecasts ====')
    nomads = 'https://nomads.ncdc.noaa.gov/data/cfsr-rfl-ts9'
    start = time.time()
    for var in cfs_vars:
        # Download the forecast of each desired variable
        url = '{}/{}/{:%Y%m}/{}.{:%Y%m%d%H}.time.grb2'.format(nomads, var, idate, var, idate)
        localfile = '{}/fcst.{}.cfsv2.{:%Y%m%d%H}.grb2'.format(cfsdir, var, idate)
        if os.path.isfile(localfile):
            if verbose: print('File already exists:\n{}'.format(localfile))
            continue
        try:
            # Download the forecast
            if verbose: print('Downloading {}...'.format(url))
            urlretrieve(url, localfile)
        except:
            if verbose: print('{} not found'.format(var))
    end = time.time()
    print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    return

####################################################################################################

def download_cfs_oper(idate, fdate, cfsdir, verbose=False, anlonly=False):
    """
    Downloads operational CFSv2 forecasts (initalized on idate) and 
    corresponding analyses (from idate to fdate)
    
    """
    from urllib.request import urlretrieve
    import time
    ###
    verbose = True
    ###
    if not os.path.isdir(cfsdir):
        os.system('mkdir {}'.format(cfsdir))
    
    #==== First download the reanalyses ========================================
    print('\n==== Downloading operational CFS analyses ====')
    nomads = 'https://nomads.ncdc.noaa.gov/modeldata/cfsv2_analysis_timeseries'
    start = time.time()
    
    # create a list of tuples specifying the years and months encompassed
    # in the desired date range
    dts = list(set([(dt.year, dt.month) for dt in [idate+timedelta(days=d) for d in \
                                                   range((fdate-idate).days + 1)]]))
    for y,m in dts:
        for var in cfs_vars:
            # Download the forecast of each desired variable
            yyyy = '{:04d}'.format(y)
            yyyymm = '{}{:02d}'.format(yyyy, m)
            url = '{}/{}/{}/{}.gdas.{}.grib2'.format(nomads, yyyy, yyyymm, var, yyyymm)
            localfile = '{}/anl.{}.cfs.{:02d}.grb2'.format(cfsdir, var, m)
            if os.path.isfile(localfile):
                if verbose: print('File already exists:\n{}'.format(localfile))
                continue
            try:
                # Download the forecast
                if verbose: print('Downloading {}...'.format(url))
                urlretrieve(url, localfile)
            except:
                if verbose: print('{} not found'.format(var))
    end = time.time()
    print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    
    # if we only want the reanalyses, then we're done!
    if anlonly: return
    
    #==== Now download the reforecasts ========================================
    print('\n==== Downloading CFSv2 reforecasts ====')
    nomads = 'https://nomads.ncdc.noaa.gov/modeldata/cfsv2_forecast_ts_9mon'
    start = time.time()
    for var in cfs_vars:
        # Download the forecast of each desired variable
        # /2011/201112/20111201/2011120100/chi200.01.2011120100.daily.grb2
        url = '{}/{:%Y}/{:%Y%m}/{:%Y%m%d}/{:%Y%m%d%H}/{}.01.{:%Y%m%d%H}.daily.grb2'
        url = url.format(nomads, idate, idate, idate, idate, var, idate)
        localfile = '{}/fcst.{}.cfsv2.{:%Y%m%d%H}.grb2'.format(cfsdir, var, idate)
        if os.path.isfile(localfile):
            if verbose: print('File already exists:\n{}'.format(localfile))
            continue
        try:
            # Download the forecast
            if verbose: print('Downloading {}...'.format(url))
            urlretrieve(url, localfile)
        except:
            if verbose: print('{} not found'.format(var))
    end = time.time()
    print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    return

####################################################################################################

def download_cfs_climo(climdir, verbose=False):
    """
    Downloads CFSR 1982-2008 calibration climatology gribs
    """
    from urllib.request import urlretrieve
    import time
    
    if not os.path.isdir(climdir):
        os.system('mkdir {}'.format(climdir))
    
    #==== First download the reanalyses ========================================
    print('\n==== Downloading CFSR calibration climatology ====')
    webpath = 'http://cfs.ncep.noaa.gov/pub/raid0/cfsv2/climo_cfsr_time/mean'
    start = time.time()
    for var in cfs_vars:
        # Download the forecast of each desired variable
        url = '{}/{}.cfsr.mean.clim.daily.1982.2010.grb2'.format(webpath, var)
        localfile = '{}/{}.clim.grb2'.format(climdir, var)
        if os.path.isfile(localfile):
            if verbose: print('File already exists:\n{}'.format(localfile))
            continue
        try:
            # Download the forecast
            if verbose: print('Downloading {}...'.format(url))
            urlretrieve(url, localfile)
        except:
            if verbose: print('{} not found'.format(var))
    end = time.time()
    print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    return

####################################################################################################

def cfs_clim_grb2nc(ncdir, nctable='cfs.table'):
    """
    Converts downloaded CFSR calibration climatology gribs to netcdf; 
    the gribs are all interpolated to a 0.5-degree lat-lon grid
    before conversion to netCDF (to ensure compatibility for combination)
    
    Requires the wgrib2 utility.
    """
    from subprocess import Popen
    import time
        
    # Point to the nc_table
    tablefile = '{}/{}'.format(ncdir, nctable)
    assert os.path.isfile(tablefile)
    
    # Use the -match keyword to select the desired dates
    with open(tablefile, "r") as tfile:
        for l, line in enumerate(tfile):
            if l==14:
                matchtag = line[1:].rstrip()
                break                   
                
    # Interpolate each grib file to a 0.5-deg lat-lon grid and then convert/append
    # to one netcdf file
    start = time.time()
    for v, var in enumerate(cfs_vars):
        print('processing grib {} of {}...'.format(v+1, len(cfs_vars)))
        grbfile = '{}/{}.clim.grb2'.format(ncdir, var)
        print(grbfile)
        # Check if this field has already been converted
        ncoutfile = '{}/clim.{}.nc'.format(ncdir, var)
        if os.path.isfile(ncoutfile):
            print('clim.{}.nc already exists!'.format(var))
            continue
        print('  interpolating...')
        interpcomm = 'wgrib2 {} {} -new_grid latlon 0:720:0.5 -90:361:0.5 temp.grb2'
        interpcomm = interpcomm.format(grbfile, matchtag)
        Popen([interpcomm], shell=True).wait()
        convertcomm = 'wgrib2 temp.grb2 -nc_table {} -netcdf {}'
        convertcomm = convertcomm.format(tablefile, ncoutfile)
        print('  converting to netcdf...')
        if os.path.isfile(ncoutfile):
            # append to the file after the first iteration
            splits = convertcomm.split('-nc_table')
            convertcomm = splits[0] + '-append -nc_table' + splits[1]
        Popen([convertcomm], shell=True).wait()
        Popen(['rm -f temp.grb2'], shell=True).wait()
    end = time.time()
    print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    return

#############################################################################################################
### SECTION FOR HANDLING GFS GLOBAL ANALYSES ################################################################
#############################################################################################################

def download_gfsanl(idate, fdate, workdir, verbose=False):
    """
    Downloads all the 3-hourly GFS analyses (via the NOMADS server) 
    from [idate] through [fdate] to the location [dpath].
    """
    ftpanldir = 'GFS/analysis_only'
    if not os.path.isdir('{}/GFS_ANL'.format(workdir)):
        os.system('mkdir {}/GFS_ANL'.format(workdir))
        
    # create a list of tuples specifying the yr, mon, and day of each day in the forecast
    dts = [(dt.year, dt.month, dt.day) for dt in [idate+timedelta(days=d) for d in \
                                                  range((fdate-idate).days + 1)]]
    # login
    ftp = FTP("nomads.ncdc.noaa.gov", "anonymous", "anonymous")
    
    # loop through all the desired days
    for dt in dts:
        y, m, d = dt
        # cd to directory
        yyyymm = '{:04d}{:02d}'.format(y, m)
        ftp.cwd("/GFS/analysis_only/{}/{}{:02d}".format(yyyymm, yyyymm, d))
        
        # which files do we want?
        files = []
        ftp.retrlines("NLST", files.append)
        for file in files:
            # We don't want any anlayses 6h after cycle init
            if file[-8:] not in ['000.grb2', '003.grb2']:
                continue
            # this is where we will download the file
            local_filename = '{}/GFS_ANL/{}'.format(workdir, file)
            
            # get datetime info for this file's analysis
            fy = int(file[9:13]); fm = int(file[13:15]); fd = int(file[15:17])
            fh = int(file[18:20]) + int(file[-7:-5])
            file_dt = datetime(fy, fm, fd, fh)
            
            # check if we already have the file
            if os.path.isfile(local_filename):
                if verbose: print('File already exists:\n{}'.format(local_filename))
            # check if this analysis is within our desired time range
            elif idate <= file_dt <= fdate:
                # DOWNLOAD!
                if verbose: print('Downloading {}...'.format(file))
                lf = open(local_filename, "wb")
                ftp.retrbinary("RETR " + file, lf.write)
                lf.close()
    ftp.close()
    return

#############################################################################################################
### SECTION FOR HANDLING SATELLITE-DERIVED PRECIPITATION ####################################################
#############################################################################################################

def download_gpm_imerg(username, idate, fdate, workdir, verbose=False):
    """
    Downloads all the 0.1deg, half-hourly GPM IMERG data (via NASA ftp) 
    from [idate] through [fdate] to the location [workdir].
    
    [username] corresponds to your registered PPM account
    
    Example file format:
    3B-HHR-L.MS.MRG.3IMERG.20170401-S000000-E002959.0000.V04A.RT-H5
    """
    from subprocess import Popen
            
    # create a list of tuples specifying the yr, mon, and day of each day in the forecast
    dts = [(dt.year, dt.month, dt.day) for dt in [idate+timedelta(days=d) for d in \
                                                  range((fdate-idate).days + 1)]]
    # our download directory:
    if not os.path.isdir('{}/GPM_IMERG'.format(workdir)):
        Popen(['mkdir {}/GPM_IMERG'.format(workdir)], shell=True).wait()
    
    # login
    ftp = FTP("jsimpson.pps.eosdis.nasa.gov", username, username)
    
    print('Downloading GPM data from {:%Y%m%d%H} through {:%Y%m%d%H}'.format(idate, fdate))
    # loop through all the desired days
    for dt in dts:
        y, m, d = dt
        # cd to directory
        yyyymm = '{:04d}{:02d}'.format(y, m)
        ftp.cwd("/NRTPUB/imerg/late/{}".format(yyyymm))
        
        # which files do we want?
        files = []
        ftp.retrlines("NLST", files.append)
        for file in files:
            
            # ignore subdirectories
            if len(file) < 40: continue
                
            # get datetime info for each file
            fd = int(file[-34:-32])
            if fd != d: continue # must be on the desired day
            hhmmss = file[-30:-24]
            fh = int(hhmmss[:2])
            fm = int(hhmmss[2:4])

            file_dt = datetime(y, m, fd, fh, fm)
            
            # this is where we will download the file
            local_filename = '{}/GPM_IMERG/{}'.format(workdir, file)
            
            # check if we already have the file
            if os.path.isfile(local_filename):
                if verbose: print('File already exists:\n{}'.format(local_filename))
            # check if this analysis is within our desired time range
            elif idate <= file_dt <= fdate:
                # DOWNLOAD!
                if verbose: print('Downloading {}...'.format(file))
                lf = open(local_filename, "wb")
                ftp.retrbinary("RETR " + file, lf.write)
                lf.close()
    ftp.close()
    return

####################################################################################################

def process_gpm_imerg(ftpusername, idate, fdate, workdir, ncfile):
    """ Loads GPM IMERG hdfs and stores as xarray/netcdf """
    from subprocess import check_output
    import h5py
    import xarray
    
    gpmdir = '{}/GPM_IMERG'.format(workdir)
    # Is the data already downloaded?
    if not os.path.isdir(gpmdir):
        # If not, download the hdfs via ftp
        print('GPM directory not found!')
        download_gpm_imerg(ftpusername, idate, fdate, workdir)
    # list the hdf files
    gpmfiles = check_output(['ls -1a {}/*H5'.format(gpmdir)], shell=True).split()

    # load the just the precipitation variable from each file
    varlist = [None] * len(gpmfiles)
    print('Reading GPM hdf5 files...')
    for f, file in enumerate(gpmfiles):
        if (f+1) % 50 == 0: print(' {} of {}'.format(f+1, len(gpmfiles)))
        # Use h5py to load the precip data
        gpmf = h5py.File(file.decode('utf8'), 'r')
        prec = gpmf['Grid/precipitationCal'][:]
        if f==0:
            lats = np.around(gpmf['Grid/lat'][:].astype(float), 2)
            lons = np.around(gpmf['Grid/lon'][:].astype(float), 2)
        gpmf.close()
        # Store the precip data as a DataArray
        varlist[f] = xarray.DataArray(prec, dims=('nLons','nLats'))

    # concatenate and rearrange the precip variable DataArrays
    precip = xarray.concat(varlist, dim='Time').transpose('Time','nLats','nLons')
    precip = precip.where(precip!=-9999.9)
    # Create lat/lon variables
    lat = xarray.DataArray(lats, dims='nLats')
    lon = xarray.DataArray(lons, dims='nLons')
    # Write to netcdf
    dset = xarray.Dataset({'precip' : precip, 'lat' : lat, 'lon' : lon})
    dset.to_netcdf('{}/{}'.format(gpmdir, ncfile))
    return precip, lat, lon

####################################################################################################

def download_trmm_3b42rt(username, idate, fdate, workdir, verbose=False):
    """
    Downloads all the 0.25deg, 3-hourly TRMM 3B42RT data (via NASA ftp) 
    from [idate] through [fdate] to the location [workdir].
    
    [username] corresponds to your registered PPM account
    
    Example file format:
    3B42.20170212.21.7.HDF.gz
    """
    from subprocess import Popen
            
    # create a list of tuples specifying the yr, mon, and day of each day in the forecast
    dts = [(dt.year, dt.month, dt.day) for dt in [idate+timedelta(days=d) for d in \
                                                  range((fdate-idate).days + 1)]]
    # our download directory:
    if not os.path.isdir('{}/TRMM_3B42RT'.format(workdir)):
        Popen(['mkdir {}/TRMM_3B42RT'.format(workdir)], shell=True).wait()
    
    # login
    ftp = FTP("arthurhou.pps.eosdis.nasa.gov", username, username)
    
    print('Downloading TRMM data from {:%Y%m%d%H} through {:%Y%m%d%H}'.format(idate, fdate))
    # loop through all the desired days
    for dt in dts:
        y, m, d = dt
        # cd to directory
        ftp.cwd("/trmmdata/ByDate/V07/{:04d}/{:02d}/{:02d}".format(y,m,d))
        
        # which files do we want?
        files = []
        ftp.retrlines("NLST", files.append)
        for file in files:
            # ignore subdirectories
            if len(file) < 25: continue
                
            # get datetime info for each file
            fh = int(file[-11:-9])

            file_dt = datetime(y, m, d, fh)
            
            # this is where we will download the file
            local_filename = '{}/TRMM_3B42RT/{}'.format(workdir, file)
            
            # check if we already have the file
            if os.path.isfile(local_filename.split('.gz')[0]):
                if verbose: print('File already exists:\n{}'.format(local_filename))
            # check if this analysis is within our desired time range
            elif idate <= file_dt <= fdate:
                # DOWNLOAD!
                if verbose: print('Downloading {}...'.format(file))
                lf = open(local_filename, "wb")
                ftp.retrbinary("RETR " + file, lf.write)
                lf.close()
                # unzip
                Popen(['gunzip {}'.format(local_filename)], shell=True).wait()
    ftp.close()
    return

#############################################################################################################
### OTHER VERIFICATION TOOLS ################################################################################
#############################################################################################################

def compute_spatial_error(field, fcst, anl, err='mae', 
                          lllat=-90, lllon=0, urlat=90, urlon=360):
    """
    Computes the mean absolute error or bias for a given field in a given domain. 
    """
    from mpasoutput import nearest_ind
    
    # Get the lat/lon indices for the desired domain
    lats, lons = fcst.latlons()
    alats, alons = anl.latlons()
    assert (np.shape(lats)==np.shape(alats))
    assert (np.shape(lons)==np.shape(alons))
    assert fcst[field].shape==anl[field].shape
    assert len(fcst[field].shape)==3
      
    # Compute the spatial error at each lead time
    # This will be SLOW unless the data is divided into chunks 
    # (see "chunks" option in MPASprocessed class, or on xarray.Dataset page)
        
    ffield = fcst.subset(field, ll=(lllat,lllon), ur=(urlat,urlon), aw=True).values
    afield = anl.subset(field, ll=(lllat,lllon), ur=(urlat,urlon), aw=True).values
    # Mean Absolute Error
    if err.lower() == 'mae':
        error = np.abs(ffield - afield).mean(axis=(1,2))
    # Bias
    elif err.lower() == 'bias':
        error =  (ffield - afield).mean(axis=(1,2))
    # Correlation
    elif err.lower() in ['corr', 'ac', 'acc']:
        raise ValueError("Haven't implemented correlation yet...")
    else:
        raise ValueError('"{}" is not a valid error parameter.'.format(err))
    
    return error

####################################################################################################

def compute_temporal_error(field, fcst, anl, err='mae', t1=None, t2=None):
    """
    Computes the mean absolute error or bias for a given field from 
    time t1 to t2 at each grid point. 
    """
    from mpasoutput import nearest_ind
    
    assert fcst[field].shape==anl[field].shape
    assert len(fcst[field].shape)==3
    
    if t1 is None or t2 is None:
        ffield = fcst[field].values
        afield = anl[field].values
    else:
        # find the desired time range
        vdates = fcst.vdates()
        i1 = nearest_ind(vdates, t1)
        i2 = nearest_ind(vdates, t2)+1
        ffield = fcst.isel(Time=range(i1,i2))[field].values
        afield = anl.isel(Time=range(i1,i2))[field].values
      
    # Compute the temporal error at each grid point
    # This will be SLOW unless the data is divided into chunks 
    # (see "chunks" option in MPASprocessed class, or on xarray.Dataset page)

    # Mean Absolute Error
    if err.lower() == 'mae':
        error =  (np.abs(ffield - afield)).mean(axis=(0))
    # Bias
    elif err.lower() == 'bias':
        error =  ((ffield - afield)).mean(axis=(0))
    # Correlation
    elif err.lower() in ['corr', 'ac', 'acc']:
        raise ValueError("Haven't implemented correlation yet...")
    else:
        raise ValueError('"{}" is not a valid error parameter.'.format(err))
    
    return error

####################################################################################################

def errormap(m, ax, cax, field, forecast, analysis, clevs, cmap=color_map('CBR_coldhot'),
             idate=None, vdate=None, units=None, cbar=True, swaplons=False):
    from copy import deepcopy
    from plotting_mpas_latlon import get_contour_levs
    import matplotlib
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    # If the projection (e.g., Mercator) crosses 180, we need to restructure the longitudes
    fcst = deepcopy(forecast)
    anl = deepcopy(analysis)
    if swaplons: 
        fcst.restructure_lons(); anl.restructure_lons()
    # Select a time if fcst_xry contains multiple valid times
    if 'Time' in fcst.dims:
        fcst = fcst.isel(Time=np.where(forecast.vdates()==vdate)[0][0])
        anl = anl.isel(Time=np.where(analysis.vdates()==vdate)[0][0])
    # mapped lat/lon locations
    x, y = fcst.project_coordinates(m)
    # Plot the fields
    cs1 = m.contour(x, y, fcst[field].values, levels=clevs, colors='darkgoldenrod', linewidths=2)
    plt.clabel(cs1, np.array(cs1.levels[::2]).astype(int), fmt='%03d', inline=1, fontsize=10)
    cs2 = m.contour(x, y, anl[field].values, levels=clevs, colors='darkgreen', linewidths=2)
    plt.clabel(cs2, np.array(cs2.levels[::2]).astype(int), fmt='%03d', inline=1, fontsize=10)
    # Plot the error
    err = fcst[field].values - anl[field].values
    flevs = get_contour_levs(err, zerocenter=True)
    csf = m.contourf(x, y, err, levels=flevs, cmap=cmap, extend='both')
    if cbar: plt.colorbar(csf, cax=cax, label='error [{}]'.format(units))
    # Set titles
    returns = [cs1, cs2, csf]
    if idate is not None and vdate is not None and units is not None:
        maintitle = 'forecast (orange) and analysis (green) {} [{}]'.format(field, units)
        ax.text(0.0, 1.015, maintitle, transform=ax.transAxes, ha='left', va='bottom', fontsize=14)
        txt = ax.text(1.0, 1.01, 'valid: {:%Y-%m-%d %H:00}'.format(vdate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
        ax.text(1.0, 1.045, 'init: {:%Y-%m-%d %H:00}'.format(idate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
        returns.append(txt)
    return returns