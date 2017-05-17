#!/usr/bin/env python
"""
Functions for downloading/processing verification dataset.

Available datasets with this module:
- GFS analyses (0.5 deg, 3-hourly)
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from ftplib import FTP
import os
from color_maker.color_maker import color_map

####################################################################################################

def download_gfsanl(idate, fdate, dpath):
    """
    Downloads all the 3-hourly GFS analyses (via the NOMADS server) 
    from [idate] through [fdate] to the location [dpath].
    """
    ftpanldir = 'GFS/analysis_only'
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
            local_filename = '{}/{}'.format(dpath, file)
            
            # get datetime info for this file's analysis
            fy = int(file[9:13]); fm = int(file[13:15]); fd = int(file[15:17])
            fh = int(file[18:20]) + int(file[-7:-5])
            file_dt = datetime(fy, fm, fd, fh)
            
            # check if we already have the file
            if os.path.isfile(local_filename):
                print('File already exists:\n{}'.format(local_filename))
            # check if this analysis is within our desired time range
            elif idate <= file_dt <= fdate:
                # DOWNLOAD!
                print('Downloading {}...'.format(file))
                lf = open(local_filename, "wb")
                ftp.retrbinary("RETR " + file, lf.write)
                lf.close()
    ftp.close()
    return

####################################################################################################

def convert_gfs_grb2nc(workdir, nctable='gfs_verification.table', outfile='gfs_analyses.nc'):
    """
    Converts downloaded GFS gribs to one netcdf file, retaining only the variables
    designated in gfs_verification.table
    
    Requires the wgrib2 utility.
    """
    from subprocess import check_output, Popen
    
    # Check if the netcdf already exists
    ncoutfile = '{}/{}'.format(workdir, outfile)
    if os.path.isfile(ncoutfile):
        print('GFS netCDF file already exists!')
        return
    
    # Point to the nc_table
    tablefile = '{}/{}'.format(workdir, nctable)
    assert os.path.isfile(tablefile)
    
    # List the gribs to be converted
    gfsgrbs = check_output(['ls -1a {}/gfsanl*.grb2'.format(workdir)], shell=True).split()
    gfsgrbs = [g.decode("utf-8") for g in gfsgrbs]
    
    # Get the -append keyword (variable info) from the nctable comments (line 15)
    with open(tablefile) as infile:
        for i, line in enumerate(infile):
            if i==14: matchtag=line[1:].rstrip(); break
                
    # Convert the gribs and append them to the same netcdf output file
    for g, grbfile in enumerate(gfsgrbs):
        print('converting grib {} of {}...'.format(g+1, len(gfsgrbs)))
        if g==0:
            wgribcommand = 'wgrib2 {} {} -nc_table {} -netcdf {}'.format(grbfile, matchtag,
                                                                 tablefile, ncoutfile)
        else:
            wgribcommand = 'wgrib2 {} {} -append -nc_table {} -netcdf {}'.format(grbfile, 
                                                       matchtag, tablefile, ncoutfile)
        Popen([wgribcommand], shell=True).wait()
    return

####################################################################################################

def compute_spatial_error(field, fcst, anl, err='mae', 
                          lllat=-90, lllon=0, urlat=90, urlon=360):
    """
    Computes the mean absolute error or bias for a given field in a given domain. 
    """
    from mpasoutput import nearest_ind
    
    # Get the lat/lon indices for the desired domain
    lats, lons = fcst.latlons()
    alats, alons = anl.latlons()
    assert (lats==alats).all()
    assert (lons==alons).all()
    if (lons < 0).any():
        raise ValueError('ERROR: longitudes must be from 0 to 360.')
    yi = nearest_ind(lats, lllat)
    yf = nearest_ind(lats, urlat)+1
    xi = nearest_ind(lons, lllon)
    xf = nearest_ind(lons, urlon)+1
            
    assert fcst[field].shape==anl[field].shape
    assert len(fcst[field].shape)==3
      
    # Compute the spatial error at each lead time
    # This will be SLOW unless the data is divided into chunks 
    # (see "chunks" option in MPASprocessed class, or on xarray.Dataset page)
        
    ffield = fcst[field].isel(nLats=range(yi,yf),nLons=range(xi,xf)).values
    afield = anl[field].isel(nLats=range(yi,yf),nLons=range(xi,xf)).values
    # Mean Absolute Error
    if err.lower() == 'mae':
        error =  (np.abs(ffield - afield) * fcst.area_weights()[None,yi:yf,None]).mean(axis=(1,2))
    # Bias
    elif err.lower() == 'bias':
        error =  ((ffield - afield) * fcst.area_weights()[None,yi:yf,None]).mean(axis=(1,2))
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