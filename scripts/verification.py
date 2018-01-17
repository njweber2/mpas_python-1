#!/usr/bin/env python
"""
Functions for downloading/processing verification datasets.

Available datasets with this module:
- CFSR reforecasts, analyses, and climatology (0.5 deg, 6-hourly)
- CFSv2 operational forecasts and analyses (0.5 deg, 6-hourly)
- GFS analyses (0.5 deg, 3-hourly)
- GPM IMERG precipitation data (0.1 deg, half-hourly)
- TRMM 3B42 precipitation data (0.25 deg, 3-hourly)
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

def remove_duplicates(values):
    """Removes duplicates from a list of values"""
    output = []
    seen = set()
    for value in values:
        # If value has not been encountered yet,
        # ... add it to both list and set.
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output

def convert_grb2nc(ncdir, nctable='cfs.table', outfile='cfsr.nc', daterange=None,
                   isanalysis=True, interp=True, gribtag=None, vrbls=None, verbose=False):
    """
    Converts downloaded NCEP gribs (forecasts or analyses) to netcdf, 
    retaining only the dates, forecast types, and variables designated 
    in nctable file and the -match keyword
    
    Optionally, the gribs are all interpolated to a 0.5-degree lat-lon grid
    before conversion to netCDF (to ensure compatibility for combination)
    
    Requires:
    *the wgrib2 utility*
    ncdir ------> directory containing the gribs files (where the netcdf will be output)
    nctable ----> name of table file used in the wgrib2 conversion command 
                  (can be either full path or just filename, assuming it is in [ncdir])
    outfile ----> name of the netcdf file to be created
    daterange --> a tuple of two datetime objects
    isanalysis -> do the gribs we are processing contain analysis data?
    interp -----> are we interpolating the data to a 0.5-deg lat-lon grid?
    gribtag ----> string at the front of the desired gribs' file names
    vrbls ------> a list of the desired variables
    """
    from subprocess import check_output, Popen
    import time
    from mpasoutput import timedelta_hours
    
    ncoutfile = '{}/{}'.format(ncdir, outfile)
    if os.path.isfile(ncoutfile):
        if verbose: print('{} already exists!'.format(outfile))
        return
    # Point to the nc_table
    if nctable[0] == '/':
        tablefile = nctable
    else:
        tablefile = '{}/{}'.format(ncdir, nctable)
    assert os.path.isfile(tablefile)
    
    # List the gribs to be converted
    if isanalysis and gribtag is None:
        gribtag = 'anl'
    elif gribtag is None:
        gribtag = 'fcst'
    # If we're processing monthly analysis files, we need to make sure we load the dates in order
    if isanalysis and 'cfs' in nctable and daterange is not None:
        grbfiles = []
        mons = remove_duplicates([dt.month for dt in [daterange[0]+timedelta(days=d) for d in \
                                  range((daterange[1]-daterange[0]).days + 1)]])
        for m in mons:
            files = check_output(['ls -1a {}/{}*{:02d}.grb2'.format(ncdir, gribtag, m)], shell=True).split()
            grbfiles += [g.decode("utf-8") for g in files]
    else:
        grbfiles = check_output(['ls -1a {}/{}*.grb2'.format(ncdir, gribtag)], shell=True).split()
        grbfiles = [g.decode("utf-8") for g in grbfiles]
        
    # Only convert the desired variables
    if vrbls is not None:
        grbfiles = [grb for grb in grbfiles if any([vrbl in grb for vrbl in vrbls])]
        
    # Our first grib file *must* be for a variable that is not temporally averaged (like precip/OLR)
    if any([vrbl in grbfiles[0] for vrbl in ['prate', 'ulwtoa']]):
        # Find the first grb with any other variable
        i = 0
        for g, grbfile in enumerate(grbfiles):
            if not any([vrbl in grbfile for vrbl in ['prate', 'ulwtoa']]):
                i = g
                break
        # Swap the first file with the non-OLR/prate file
        grbfiles[0], grbfiles[i] = grbfiles[i], grbfiles[0]
    
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
        if verbose: print('processing grib {} of {}...'.format(g+1, len(grbfiles)))
        if verbose: print(grbfile)
        if interp:
            if verbose: print('  interpolating...')
            interpcomm = 'wgrib2 {} {} -new_grid_vectors U:V -new_grid latlon 0:720:0.5 -90:361:0.5 temp.grb2'
            interpcomm = interpcomm.format(grbfile, matchtag)
            Popen([interpcomm], shell=True).wait()
            convertcomm = 'wgrib2 temp.grb2 -nc_table {} -netcdf {}'
            convertcomm = convertcomm.format(tablefile, ncoutfile)
        else:
            convertcomm = 'wgrib2 {} {} -nc_table {} -netcdf {}'
            convertcomm = convertcomm.format(grbfile, matchtag, tablefile, ncoutfile)
        if verbose: print('  converting to netcdf...')
        if os.path.isfile(ncoutfile):
            splits = convertcomm.split('-nc_table')
            convertcomm = splits[0] + '-append -nc_table' + splits[1]
        Popen([convertcomm], shell=True).wait()
        if interp: Popen(['rm -f temp.grb2'], shell=True).wait()
    end = time.time()
    if verbose: print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    return

#############################################################################################################
### SECTION FOR HANDLING OPERATIONAL/RETROSPECTIVE CFSv2 FORECASTS/ANALYSES #################################
#############################################################################################################
cfs_vars = ['chi200', 'prate', 'pressfc', 'psi200', 'pwat', 'tmp2m',
            'tmpsfc', 'ulwtoa', 'wnd200', 'wnd850', 'z500', 'z200', 'lhtfl']

def download_cfsrr(idate, fdate, cfsdir, vrbls=None, anlonly=False, verbose=False):
    """
    Downloads CFSv2 reforecasts (initalized on idate) and 
    corresponding analyses (CFSR; from idate to fdate)
    
    Requires:
    idate ---> initialization date (datetime)
    fdate ---> final/end date (datetime)
    cfsdir --> full path to CFS download location
    vrbls ---> list of desired variables
    anlonly -> download analyses only, and not the reforecasts?
    """
    from urllib.request import urlretrieve
    import time
    
    if not os.path.isdir(cfsdir):
        os.system('mkdir {}'.format(cfsdir))
    
    #==== First download the reanalyses ========================================
    if verbose: print('\n==== Downloading CFSR reanalyses ====')
    nomads = 'https://nomads.ncdc.noaa.gov/data/cfsr'
    start = time.time()
    
    # create a list of tuples specifying the years and months encompassed
    # in the desired date range
    dts = list(set([(dt.year, dt.month) for dt in [idate+timedelta(days=d) for d in \
                                                   range((fdate-idate).days + 1)]]))
    if vrbls is None:
        vrbls = cfs_vars

    for y,m in dts:
        for var in vrbls:
            # Download the analysis of each desired variable
            yyyymm = '{:04d}{:02d}'.format(y, m)
            url = '{}/{}/{}.gdas.{}.grb2'.format(nomads, yyyymm, var, yyyymm)
            localfile = '{}/anl.{}.cfs.{}.grb2'.format(cfsdir, var, yyyymm)
            if os.path.isfile(localfile):
                if verbose: print('File already exists:\n{}'.format(localfile))
                continue
            try:
                # Download the analysis
                if verbose: print('Downloading {}...'.format(url))
                urlretrieve(url, localfile)
            except:
                if verbose: print('{} not found'.format(var))
    end = time.time()
    if verbose: print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    
    # if we only want the reanalyses, then we're done!
    if anlonly: return
    
    #==== Now download the reforecasts ========================================
    if verbose: print('\n==== Downloading CFSv2 reforecasts ====')
    nomads = 'https://nomads.ncdc.noaa.gov/data/cfsr-rfl-ts9'
    start = time.time()
    for var in vrbls:
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
    if verbose: print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    return

####################################################################################################

def download_cfs_oper(idate, fdate, cfsdir, vrbls=None, anlonly=False, verbose=False):
    """
    Downloads operational CFSv2 forecasts (initalized on idate) and 
    corresponding analyses (from idate to fdate)
    
    Requires:
    idate ---> initialization date (datetime)
    fdate ---> final/end date (datetime)
    cfsdir --> full path to CFS download location
    vrbls ---> list of desired variables
    anlonly -> download analyses only, and not the forecasts?
    """
    from urllib.request import urlretrieve
    import time

    if not os.path.isdir(cfsdir):
        os.system('mkdir {}'.format(cfsdir))
    
    #==== First download the reanalyses ========================================
    if verbose: print('\n==== Downloading operational CFS analyses ====')
    nomads = 'https://nomads.ncdc.noaa.gov/modeldata/cfsv2_analysis_timeseries'
    start = time.time()
    
    # create a list of tuples specifying the years and months encompassed
    # in the desired date range
    dts = remove_duplicates([(dt.year, dt.month) for dt in [idate+timedelta(days=d) for d in \
                             range((fdate-idate).days + 1)]])
    if vrbls is None:
        vrbls = cfs_vars
        
    for y,m in dts:
        for var in vrbls:
            # Download the analysis of each desired variable
            yyyy = '{:04d}'.format(y)
            yyyymm = '{}{:02d}'.format(yyyy, m)
            url = '{}/{}/{}/{}.gdas.{}.grib2'.format(nomads, yyyy, yyyymm, var, yyyymm)
            localfile = '{}/anl.{}.cfs.{:02d}.grb2'.format(cfsdir, var, m)
            if os.path.isfile(localfile):
                if verbose: print('File already exists:\n{}'.format(localfile))
                continue
            try:
                # Download the analysis
                if verbose: print('Downloading {}...'.format(url))
                urlretrieve(url, localfile)
            except:
                if verbose: print('{} not found'.format(var))
    end = time.time()
    if verbose: print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    
    # if we only want the reanalyses, then we're done!
    if anlonly: return
    
    #==== Now download the reforecasts ========================================
    if verbose: print('\n==== Downloading CFSv2 reforecasts ====')
    nomads = 'https://nomads.ncdc.noaa.gov/modeldata/cfsv2_forecast_ts_9mon'
    start = time.time()
    for var in vrbls:
        # Download the forecast of each desired variable
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
    if verbose: print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    return

####################################################################################################

def download_cfs_climo(climdir, verbose=False):
    """
    Downloads CFSR 1982-2008 calibration climatology gribs
    
    Requires:
    climdir --> full path to CFS download location
    """
    from urllib.request import urlretrieve
    import time
    
    if not os.path.isdir(climdir):
        os.system('mkdir {}'.format(climdir))
    
    if verbose: print('\n==== Downloading CFSR calibration climatology ====')
    webpath = 'http://cfs.ncep.noaa.gov/pub/raid0/cfsv2/climo_cfsr_time/mean'
    start = time.time()
    for var in cfs_vars:
        # Download the file for each desired variable
        url = '{}/{}.cfsr.mean.clim.daily.1982.2010.grb2'.format(webpath, var)
        localfile = '{}/{}.clim.grb2'.format(climdir, var)
        if os.path.isfile(localfile):
            if verbose: print('File already exists:\n{}'.format(localfile))
            continue
        try:
            # Download the file
            if verbose: print('Downloading {}...'.format(url))
            urlretrieve(url, localfile)
        except:
            if verbose: print('{} not found'.format(var))
    end = time.time()
    if verbose: print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    return

####################################################################################################

def cfs_clim_grb2nc(ncdir, nctable='cfs.table', verbose=False):
    """
    Converts downloaded CFSR calibration climatology gribs to netcdf; 
    the gribs are all interpolated to a 0.5-degree lat-lon grid
    before conversion to netCDF (to ensure compatibility for combination)
    
    Requires:
    *the wgrib2 utility(
    ncdir ---> directory containing the gribs files (where the netcdf will be output)
    nctable -> name of table file used in the wgrib2 conversion command 
               (can be either full path or just filename, assuming it is in [ncdir])
    """
    from subprocess import Popen
    import time
        
    # Point to the nc_table
    if nctable[0] == '/':
        tablefile = nctable
    else:
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
        if verbose: print('processing grib {} of {}...'.format(v+1, len(cfs_vars)))
        grbfile = '{}/{}.clim.grb2'.format(ncdir, var)
        if verbose: print(grbfile)
        # Check if this field has already been converted
        ncoutfile = '{}/clim.{}.nc'.format(ncdir, var)
        if os.path.isfile(ncoutfile):
            if verbose: print('clim.{}.nc already exists!'.format(var))
            continue
        if verbose: print('  interpolating...')
        interpcomm = 'wgrib2 {} {} -new_grid latlon 0:720:0.5 -90:361:0.5 temp.grb2'
        interpcomm = interpcomm.format(grbfile, matchtag)
        Popen([interpcomm], shell=True).wait()
        convertcomm = 'wgrib2 temp.grb2 -nc_table {} -netcdf {}'
        convertcomm = convertcomm.format(tablefile, ncoutfile)
        if verbose: print('  converting to netcdf...')
        if os.path.isfile(ncoutfile):
            # append to the file after the first iteration
            splits = convertcomm.split('-nc_table')
            convertcomm = splits[0] + '-append -nc_table' + splits[1]
        Popen([convertcomm], shell=True).wait()
        Popen(['rm -f temp.grb2'], shell=True).wait()
    end = time.time()
    if verbose: print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    return

#############################################################################################################
### SECTION FOR HANDLING GFS GLOBAL ANALYSES ################################################################
#############################################################################################################

def download_gfsanl(idate, fdate, workdir, verbose=False):
    """
    Downloads all the 3-hourly GFS analyses (via the NOMADS server) 
    from [idate] through [fdate] to the location [workdir]/GFSANL.
    
    Requires:
    idate ---> initialization date (datetime)
    fdate ---> final/end date (datetime)
    workdir -> full path to working directory
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
    from [idate] through [fdate] to the location [workdir]/GPM_IMERG.
    
    Example file format:
    3B-HHR-L.MS.MRG.3IMERG.20170401-S000000-E002959.0000.V04A.RT-H5
    
    Requires:
    username -> username for a registered PPM account
    idate ----> initialization date (datetime)
    fdate ----> final/end date (datetime)
    workdir --> full path to working directory
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
    
    if verbose: print('Downloading GPM data from {:%Y%m%d%H} through {:%Y%m%d%H}'.format(idate, fdate))
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

def process_gpm_imerg(ftpusername, idate, fdate, workdir, ncfile, verbose=False):
    """
    Loads GPM IMERG hdfs and stores as xarray/netcdf
    
    Requires:
    ftpusername -> username for a registered PPM account
    idate -------> initialization date (datetime)
    fdate -------> final/end date (datetime)
    workdir -----> full path to working directory
    ncfile ------> name of netcdf file (in [workdir]/GPM_IMERG) to store GPM data in
    
    Returns:
    precip ------> xarray DataArray of precipitation data
    lat ---------> xarray DataArray of latitudes
    lon ---------> xarray DataArray of longitudes
    """
    from subprocess import check_output
    import h5py
    import xarray
    
    gpmdir = '{}/GPM_IMERG'.format(workdir)
    # Is the data already downloaded?
    if not os.path.isdir(gpmdir):
        # If not, download the hdfs via ftp
        if verbose: print('GPM directory not found!')
        download_gpm_imerg(ftpusername, idate, fdate, workdir, verbose=verbose)
    # list the hdf files
    gpmfiles = check_output(['ls -1a {}/*H5'.format(gpmdir)], shell=True).split()

    # load just the precipitation variable from each file
    varlist = [None] * len(gpmfiles)
    if verbose: print('Reading GPM hdf5 files...')
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
    
    Example file format:
    3B42.20170212.21.7.HDF.gz
    
    Requires:
    username -> username for a registered PPM account
    idate ----> initialization date (datetime)
    fdate ----> final/end date (datetime)
    workdir --> full path to working directory
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
    
    if verbose: print('Downloading TRMM data from {:%Y%m%d%H} through {:%Y%m%d%H}'.format(idate, fdate))
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
            if file[:4] != '3B42': continue
                
            # get datetime info for each file
            fh = int(file[14:16])
            if verbose: print(fh, file)
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

####################################################################################################

def process_ssmi_raw(startdate, ndays, ssmidir, vrbls=['vapor', 'rain'], missing=-999.,
                     satellites=['f16','f17'], verbose=False):
    """
    Converts raw, gzipped daily SSMI data into grouped daily netcdf files
    
    Requires:
    startdate --> initialization date (datetime)
    ndays ------> number of days' worth of SSMI data
    ssmidir ----> full path to directory with the raw data (will also be the nc output directory)
    vrbls ------> list of variables to extract from the SSMI files
    missing ----> value for missing data
    satellites -> list of satellites to extract the data from (f[xx])
    """
    from datatools.ssmi_daily_v7 import SSMIdaily
    from netCDF4 import Dataset, num2date, date2num

    # Name of the netcdf file that will be created
    ncoutfile = '{}/ssmi_{}_{:%Y%m%d}-{:%Y%m%d}.nc'.format(ssmidir, ('').join(satellites), 
                                                           startdate, startdate+timedelta(days=ndays-1))
    # A dictionary to convert SSMI vrbl names to MPAS vrbl names
    vardict = {'vapor' : 'precipw', 'rain' : 'prate1h'}
    
    # Create empty full data arrays
    fulldata = {}
    for vrbl in vrbls:
        nsats = len(satellites)
        fulldata[vrbl] = np.zeros((ndays, 2*nsats, 720, 1440))  # 2 --> ascending and descending passes
        
    if verbose: print('Loading SSMI data....')
    # Load the data for each daily
    for day in range(ndays):
        date = startdate + timedelta(days=day)
        if verbose: print(date)
        # Load data from each of the satellites
        for s, satellite in enumerate(satellites):
            ssmifile = '{}/{}_{:%Y%m%d}v7.gz'.format(ssmidir, satellite, date)
            # LOAD THE DATASET
            dataset = SSMIdaily(ssmifile, missing=missing)
            # Load the dimensions
            if s==0 and day==0:
                lats = np.array(dataset.variables['latitude'])
                lons = np.array(dataset.variables['longitude'])
            i = s*2 # index for the second dimension of the full array
            # Load the data for the desired variables
            for vrbl in vrbls:
                fulldata[vrbl][day, i:i+2, :, :] = dataset.variables[vrbl]
    # Put nans in for missing data
    for vrbl in vrbls:
        fulldata[vrbl][fulldata[vrbl]==missing] = np.nan
        
    if verbose: print('Saving data to netcdf file...')
    with Dataset(ncoutfile, 'w') as ncdata:
        # Create dimensions
        ncdata.createDimension('Time', ndays)
        ncdata.createDimension('Pass', 4)
        ncdata.createDimension('nLats', len(lats))
        ncdata.createDimension('nLons', len(lons))
        # Store dimensional variables
        times = ncdata.createVariable('date', 'i', ('Time',))
        times.units = 'hours since 1800-01-01'
        times[:] = date2num(np.array([startdate+timedelta(days=d) for d in range(ndays)]), 'hours since 1800-01-01')
        passes = ncdata.createVariable('passnumber', 'i', ('Pass',))
        passes[:] = np.arange(4)+1
        la = ncdata.createVariable('lat', 'f4', ('nLats',))
        la[:] = lats
        lo = ncdata.createVariable('lon', 'f4', ('nLons',))
        lo[:] = lons
        # Store the precip and pwat data
        for vrbl in vrbls:
            var = ncdata.createVariable(vardict[vrbl], 'f4', ('Time', 'Pass', 'nLats', 'nLons',))
            var[:] = fulldata[vrbl]
    if verbose: print('Done!')

#############################################################################################################
### OTHER VERIFICATION TOOLS ################################################################################
#############################################################################################################

def compute_spatial_error(field, fcst, anl, err='mae', lllat=-90, lllon=0, urlat=90, urlon=360,
                          idate=None, fdate=None):
    """
    Computes the mean absolute error or bias for a given field in a given domain.
    
    Requires:
    field --> name of the variable (string)
    fcst ---> a LatLonData object of the forecast data
    anl ----> a LatLonData object of the analyis data
    err ----> name of the error metric to be calculated (string)
    lllat --> the latitude at the lower-left corner of the desired domain
    lllon --> the longitude at the lower-left corner of the desired domain
    urlat --> the latitude at the upper-right corner of the desired domain
    urlon --> the longitude at the upper-right corner of the desired domain
    """
    from mpasoutput import nearest_ind
    
    # Get the lat/lon indices for the desired domain
    lats, lons = fcst.latlons()
    alats, alons = anl.latlons()
    assert (np.shape(lats)==np.shape(alats))
    assert (np.shape(lons)==np.shape(alons))
    assert len(fcst[field].shape)==3
      
    # Compute the spatial error at each lead time
    # This will be SLOW unless the data is divided into chunks 
    # (see "chunks" option in LatLonData class, or on xarray.Dataset page)
    if idate is not None and fdate is not None:
        fcst = fcst.isel(Time=np.where((fcst.vdates()>=idate)*(fcst.vdates()<=fdate))[0])
        anl = anl.isel(Time=np.where((anl.vdates()>=idate)*(anl.vdates()<=fdate))[0])
    assert fcst[field].shape==anl[field].shape
        
    ffield, weights = fcst.subset(field, ll=(lllat,lllon), ur=(urlat,urlon), aw=True)
    afield, weights = anl.subset(field, ll=(lllat,lllon), ur=(urlat,urlon), aw=True)
    wgts = np.tile(weights[:,None],(np.shape(ffield)[0],1,np.shape(ffield)[-1]))
    # Mean Absolute Error
    if err.lower() == 'mae':
        diff = np.abs(ffield.values - afield.values)
        diffma = np.ma.masked_array(diff,np.isnan(diff))
        error = np.average(diffma, axis=(1,2), weights=wgts)
    # Bias
    elif err.lower() == 'bias':
        diff = ffield.values - afield.values
        diffma = np.ma.masked_array(diff,np.isnan(diff))
        error = np.average(diffma, axis=(1,2), weights=wgts)
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
    
    Requires:
    field --> name of the variable (string)
    fcst ---> a LatLonData object of the forecast data
    anl ----> a LatLonData object of the analyis data
    err ----> name of the error metric to be calculated (string)
    t1 -----> datetime object for the initial time
    t2 -----> datetime object for the final time
    """
    from mpasoutput import nearest_ind
    
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
    assert np.shape(ffield)==np.shape(afield)
      
    # Compute the temporal error at each grid point
    # This will be SLOW unless the data is divided into chunks 
    # (see "chunks" option in LatLonData class, or on xarray.Dataset page)

    # Mean Absolute Error
    if err.lower() == 'mae':
        error =  np.nanmean(np.abs(ffield - afield), axis=(0))
    # Bias
    elif err.lower() == 'bias':
        error =  np.nanmean((ffield - afield), axis=(0))
    # Correlation
    elif err.lower() in ['corr', 'ac', 'acc']:
        raise ValueError("Haven't implemented correlation yet...")
    else:
        raise ValueError('"{}" is not a valid error parameter.'.format(err))
    
    return error

####################################################################################################

def errormap(m, ax, cax, field, forecast, analysis, clevs, cmap=color_map('CBR_coldhot'),
             idate=None, vdate=None, units=None, cbar=True, swaplons=False):
    """
    Plots a contour map of a forecast and analysis field, and a contour fill of the 
    difference (error) between the two.
    
    Requires:
    m, ax, cax -> Basemap, axis, and colorbar axis objects
    field ------> the name of the desired variable to plot (string)
    forecast ---> a LatLonData object of the forecast data
    analysis ---> a LatLonData object of the analysis data
    clevs ------> list of contour levels
    cmap -------> colormap for the error contour fill
    idate ------> datetime of the forecast initialization date
    vdate ------> datetime of the desired valid date in this plot
    units ------> units of the desired [field] (string)
    cbar -------> plot the colorbar?
    swaplons ---> restructure the data to have longitudes from 0 to 360?
    
    Returns:
    cs1, cs2 ---> contour objects for the forecast and analysis
    csf --------> contour fill object for the error
    txt --------> (optionally) the title text object
    """
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
    if 'Time' in fcst.dims and vdate is not None:
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