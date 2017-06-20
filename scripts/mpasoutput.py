#!/usr/bin/env python
import numpy as np
import xarray
from datetime import datetime, timedelta
from copy import deepcopy
import os

################################################################################################
################################################################################################
# MPAS forecast (or other data) on a lat-lon grid
################################################################################################
################################################################################################

class LatLonData(xarray.Dataset):
    """
    Define a multivariate Dataset composed of MPAS forecast output or other geophysical
    data on a lat/lon grid.
    """
      
    # THE FOLLOWING ARE SEVERAL CLASS METHODS USED TO INITIALIZE A LatLonData
    # OBJECT FROM DIFFERENT DATASETS. THE IDEA IS TO HAVE ALL GRIDS (MPAS FORECASTS,
    # ANALYSIS GRIDS, OTHER MODEL FORECASTS, SATELLITE OBSERVATIONS, ETC.) STORED
    # IN THE SAME OBJECT TYPE FOR SLICK AND EASY MANIPULATION/COMPARISON
    
    @classmethod
    def from_netcdf(cls, ncfile, workdir, idate, dt, chunks=None):
        """
        Initializes a new LatLonData object when given
        a processed MPAS output file (netcdf)
        
        MPAS files are assumed to be on a lat lon grid; i.e., raw MPAS diag.nc (or any 
        other stream) files converted with the convert_mpas utility:
        https://github.com/mgduda/convert_mpas/
        
        Requires:
        ncfile ---> string: input netcdf filename
        workdir --> string: path to the input/ouput directory
        idate ----> datetime object indicating the data start date
        dt -------> the number of hours between each data time
        chunks ---> dictionary for chunking the xarray contents to dask arrays
        """
        forecast = xarray.open_dataset(ncfile, chunks=chunks)
        forecast.attrs.update(idate=idate, dt=dt, type='MPAS', workdir=workdir)
        forecast.__class__ = cls
        return forecast
    
    @classmethod
    def from_latlon_grid(cls, lats, lons, workdir, idate, dt, inputstream='diag',
                         meshinfofile='*init.nc', outputfile='diags_interp.nc', chunks=None):
        """
        Initializes a new LatLonData object when given
        a desired lat/lon grid and MPAS output stream name
        
        raw MPAS diag.nc data are interpolated to a regular
        lat/lon grid using the convert_mpas utility:
        https://github.com/mgduda/convert_mpas/
        
        Requires:
        lats, lons --> 1D numpy arrays of latitudes and longitudes
        workdir -----> string: path to the input/output directory
        idate -------> datetime object indicating the data start date
        dt ----------> the number of hours between each data time
        inputstream -> string: the MPAS output stream to be loaded (e.g., "diag" --> "diag.*.nc"
        meshinfofile > a filename (or wildcard) pointing to a netcdf with mesh info variables
        outputfile --> the name of the final (interpolated) netcdf file
        chunks ------> dictionary for chunking the xarray contents to dask arrays
        """
        from subprocess import Popen
        import time
        from xarray import DataArray
        
        # We're going to assume that, if latlon.nc does exist, it was created via this function
        # and has an identical grid to the on passed to this function
        if not os.path.isfile('{}/{}'.format(workdir, outputfile)):
            # First check if the lat/lons are regularly spaced
            dlat = (lats - np.roll(lats, 1))[1:]
            dlon = (lons - np.roll(lons, 1))[1:]
            assert (dlat==dlat[0]).all() and (dlon==dlon[0]).all()

            # Next, specify the desired grid in the "target_domain" text file
            with open('{}/target_domain'.format(workdir) ,'w') as target:
                target.write('nlat={}\n'.format(len(lats)))
                target.write('nlon={}\n'.format(len(lons)))
                target.write('startlat={:.01f}\n'.format(lats[0]-0.25))
                target.write('startlon={:.01f}\n'.format(lons[0]-0.25))
                target.write('endlat={:.01f}\n'.format(lats[-1]+0.25))
                target.write('endlon={:.01f}\n'.format(lons[-1]+0.25))

            # Now build and execute our convert_mpas command
            if os.path.isfile('{}/latlon.nc'.format(workdir)):
                Popen(['rm -f {}/latlon.nc'.format(workdir)], shell=True).wait()
            print('Interpolating {} to regularly-spaced lat-lon grid...'.format(inputstream))
            start = time.time()
            command = 'cd {}; ./convert_mpas {} {}.*.nc'.format(workdir,meshinfofile,inputstream)
            Popen([command], shell=True).wait()   # will generate a latlon.nc file
            end = time.time()
            print('Elapsed time: {:.2f} min'.format((end-start)/60.))
            # rename the output file
            Popen(['cd {}; mv latlon.nc {}'.format(workdir, outputfile)], shell=True).wait()
        
        # Finally, create our LatLonData object via xarray
        forecast = xarray.open_dataset('{}/{}'.format(workdir, outputfile), chunks=chunks)
        forecast.attrs.update(idate=idate, dt=dt, type='MPAS', workdir=workdir)
        forecast.__class__ = cls
        return forecast

    @classmethod
    def from_GFS_netcdf(cls, workdir, idate, fdate, ncfile='gfs_analyses.nc',
                        nctable='gfs.table', chunks=None):
        """
        Initializes a new LatLonData object when given
        a 3-hourly GFS analysis file (netcdf)
        
        Requires:
        workdir --> string: path to the input/output directory
        idate ----> datetime object indicating the data start date
        fdate ----> datetime object indicating the data end date
        ncfile ---> string: netcdf of GFS analyses to be loaded via xarray
        nctable --> string: text file (.table) used for grb2nc conversion
        chunks ---> dictionary for chunking the xarray contents to dask arrays
        """
        import verification as verf
        infile = '{}/GFS_ANL/{}'.format(workdir, ncfile)
        # Has the netcdf file already been created?
        if not os.path.isfile(infile):
            # If not, download the gribs via ftp and convert them to netcdf
            print('File {} not found!'.format(infile))
            verf.download_gfsanl(idate, fdate, workdir)
            verf.convert_grb2nc('{}/GFS_ANL'.format(workdir), nctable=nctable, outfile=ncfile)
        # Load the netcdf as an xarray Dataset
        analyses = xarray.open_dataset(infile, chunks=chunks)
        analyses.attrs.update(idate=idate, dt=3, type='GFS', workdir=workdir)
        analyses.__class__ = cls
        # Rename the coordinates/dims so the functions below still work
        analyses.rename({'latitude' : 'nLats', 'longitude' : 'nLons', 'time' : 'Time'}, inplace=True)
        analyses.update(analyses.assign(lat=analyses.variables['nLats']))
        analyses.update(analyses.assign(lon=analyses.variables['nLons']))
        return analyses
    
    @classmethod
    def from_TRMM_hdfs(cls, workdir, idate, fdate, ftpusername):
        """
        Initializes a new LatLonData object from a directory of
        3-hourly TRMM 3B42RT precipitation hdf files
        
        Requires:
        workdir -----> string: path to the input/output directory
        idate -------> datetime object indicating the data start date
        fdate -------> datetime object indicating the data end date
        ftpusername -> string: PPS email account for downloading TRMM/GPM data via ftp
        """
        from subprocess import check_output
        import verification as verf
        
        trmmdir = '{}/TRMM_3B42RT'.format(workdir)
        
        # Is the data already downloaded?
        try:
            trmmfiles = check_output(['ls -1a {}/*.HDF'.format(trmmdir)], shell=True).split()
        except:
            # If not, download the hdfs via ftp
            print('TRMM directory not found!')
            verf.download_trmm_3b42rt(ftpusername, idate, fdate, workdir)
            trmmfiles = check_output(['ls -1a {}/*.HDF'.format(trmmdir)], shell=True).split()
        
        # load the just the precipitation variable from each file
        varlist = []
        for f, file in enumerate(trmmfiles):
            trmmxry = xarray.open_dataset(file.decode('utf8')).rename({'nlat' : 'nLats', 'nlon' : 'nLons'})
            varlist.append(trmmxry.data_vars['precipitation'])
        # concatenate and rearrange the precip variable DataArrays
        precip = xarray.concat(varlist, dim='Time').transpose('Time','nLats','nLons')
        precip = precip.where(precip!=-9999.9)
        # The lat and lon should be calculated manually.
        # More information can be found at:
        # http://disc.sci.gsfc.nasa.gov/additional/faq/precipitation_faq.shtml#lat_lon
        lat = xarray.DataArray(np.arange(-49.875, 49.876, 0.25), dims='nLats')
        lon = xarray.DataArray(np.arange(-179.875, 179.876, 0.25), dims='nLons')

        # Create an xarray Dataset and compute 3-hourly precipitation
        trmm = xarray.Dataset({'preciprate' : precip, 'lat' : lat, 'lon' : lon})
        prate1h = trmm['preciprate'].shift(Time=1)
        prate3h = prate1h*3
        trmm.update(trmm.assign(prate1h=prate1h, prate3h=prate3h))
        trmm.attrs.update(idate=idate, dt=3, type='TRMM', workdir=workdir)
        trmm.__class__ = cls
        return trmm

    @classmethod
    def from_GPM_hdfs(cls, workdir, idate, fdate, ftpusername, ncfile='gpm.nc'):
        """
        Initializes a new LatLonData object from a directory of
        half-hourly GPM IMERG precipitation hdf files
        
        Requires:
        workdir -----> string: path to the input/output directory
        idate -------> datetime object indicating the data start date
        fdate -------> datetime object indicating the data end date
        ftpusername -> string: PPS email account for downloading TRMM/GPM data via ftp
        ncfile ------> string: netcdf file of processed IMERG data
        """
        import verification as verf
        import h5py
        
        gpmdir = '{}/GPM_IMERG'.format(workdir)
        
        # Do we need to process the HDF data?
        if not os.path.isfile('{}/{}'.format(gpmdir,ncfile)):
            precip, lat, lon = verf.process_gpm_imerg(ftpusername, idate, fdate, 
                                                      workdir, ncfile)
        # otherwise, load from pre-processed netcdf
        else:
            print('Loading from {}/{}'.format(gpmdir, ncfile))
            dset = xarray.open_dataset('{}/{}'.format(gpmdir, ncfile))
            precip = dset['precip']
            lat = dset['lat']
            lon = dset['lon']
            
        # compute hourly and 3-hourly precipitation
        prate1h = (precip.shift(Time=1)*0.5 + precip.shift(Time=2)*0.5)[::2,:,:]
        prate3h = prate1h + prate1h.shift(Time=1) + prate1h.shift(Time=2)

        # Create an xarray Dataset with the above variables
        gpm = xarray.Dataset({'prate1h' : prate1h, 'prate3h' : prate3h,
                              'lat' : lat, 'lon' : lon})
        gpm.attrs.update(idate=idate, dt=1, type='GPM', workdir=workdir)
        gpm.__class__ = cls
        return gpm
    
    @classmethod
    def from_CFSRR_netcdfs(cls, workdir, idate, fdate, nctable='cfs.table', 
                           anlonly=False, chunks=None, vrbls=None):
        """
        Initializes two new LatLonData objects (analyses and forecasts) when pointed
        to the 6-hourly CFSR/CFSv2 netcdfs

        Requires:
        workdir --> string: path to the input/output directory
        idate ----> datetime object indicating the data start date
        fdate ----> datetime object indicating the data end date
        nctable --> string: text file (.table) used for grb2nc conversion
        anlonly --> bool: are we processing ONLY analysis (CFSR) data?
        chunks ---> dictionary for chunking the xarray contents to dask arrays
        vrbls ----> list of variables to process (if None, all will be processed)
        """
        import verification as verf
        cfsdir = '{}/CFS'.format(workdir)
        anlfile = 'cfsr_{:%Y%m%d%H}-{:%Y%m%d%H}.nc'.format(idate, fdate)
        fcstfile = 'cfsv2_{:%Y%m%d%H}-{:%Y%m%d%H}.nc'.format(idate, fdate)
        anlpath = '{}/{}'.format(cfsdir, anlfile)
        fcstpath = '{}/{}'.format(cfsdir, fcstfile)

        # Have the forecast/analysis netcdf files already been created?
        if not os.path.isfile(anlpath) or not os.path.isfile(fcstpath):
            print('File(s) {} and/or {} not found in working directory!'.format(anlfile, fcstfile))
            # If the netcdfs don't exist, then download and convert the gribs
            if idate.year >= 2011:
                verf.download_cfs_oper(idate, fdate, cfsdir, vrbls=vrbls, anlonly=anlonly)
            else:
                verf.download_cfsrr(idate, fdate, cfsdir, vrbls=vrbls, anlonly=anlonly)
            print('\n==== Converting CFSR to netcdf ====')
            verf.convert_grb2nc(cfsdir, outfile=anlfile, daterange=(idate,fdate), isanalysis=True, vrbls=vrbls)
            if not anlonly:
                print('\n==== Converting CFSv2 to netcdf ====')
                verf.convert_grb2nc(cfsdir, outfile=fcstfile, daterange=(idate,fdate), isanalysis=False, vrbls=vrbls)

        # OK, we have the data downloaded, interpolated to 0.5deg latlon,
        # and converted to two netcdfs (analyses and forecast)
        # Now we just need to load them as xarrays and prepend t=0 from the 
        # analyses to the forecasts (because NCEP doesn't include the analyses in 
        # the forecast gribs.... grr. )
        print('Loading CFSR/CFSv2 from netcdfs...')
        # Process the analyses first
        analyses = xarray.open_dataset(anlpath)###########, chunks=chunks)
        # Rename the dimensions
        analyses.rename({'latitude' : 'nLats', 'longitude' : 'nLons', 'time' : 'Time'}, inplace=True)
        if vrbls is None or 'prate' in vrbls or 'ulwtoa' in vrbls:
            # Trim the last time off of the analyses (precip)
            analyses = analyses.isel(Time=range(analyses.dims['Time']-1))
        datasets = [analyses]
        types = ['CFSR']
        
        # Now process the forecast
        if not anlonly:
            forecast = xarray.open_dataset(fcstpath, chunks=chunks)
            forecast.rename({'latitude' : 'nLats', 'longitude' : 'nLons', 'time' : 'Time'}, inplace=True)
            # Prepend the t=0 analysis to the forecast (the initialization)
            prepend = analyses.isel(Time=[0]).drop([var for var in analyses.variables.keys() if var not in forecast.variables.keys()])
            forecast = xarray.concat([prepend, forecast], dim='Time')
            datasets.append(forecast)
            types.append('CFSv2')
 
        # Need to store date info, because it is not stored in the MPAS netcdf
        # metadata by default
        for dataset, type in zip(datasets, types):
            dataset.attrs.update(idate=idate, dt=6, type=type, workdir=workdir)
            dataset.__class__ = cls # Change the class to LatLonData
            # Add 'lat' and 'lon' variables so functions below still work
            dataset.update(dataset.assign(lat=analyses.variables['nLats']))
            dataset.update(dataset.assign(lon=analyses.variables['nLons']))
            # Change precipitation from mm/s to mm/h
            if 'prate1h' in dataset.variables.keys():
                dataset['prate1h'] *= 3600.
                dataset.update(dataset.assign(prate1d=dataset.variables['prate1h']*24.))
                
        if anlonly: return analyses
        else:       return analyses, forecast
        
    @classmethod
    def from_CFSclimo_netcdfs(cls, climdir, idate, fdate, nctable='cfs.table', chunks=None):
        """
        Initializes a new LatLonData object from 6-hourly CFSR calibration climatology data

        Requires:
        climdir --> string: path to the directory with the climatology data files
        idate ----> datetime object indicating the data start date
        fdate ----> datetime object indicating the data end date
        nctable --> string: text file (.table) used for grb2nc conversion
        chunks ---> dictionary for chunking the xarray contents to dask arrays
        """
        import verification as verf
        from subprocess import check_output

        climfiles = '{}/clim.*.nc'.format(climdir)

        # Have the forecast/analysis netcdf files already been created?
        try:
            filelist = check_output(['ls -1a {}'.format(climfiles)], shell=True).split()
        except:
            print('No netcdfs found in {}!'.format(climdir))
            # If the netcdfs don't exist, then download and convert the gribs
            verf.download_cfs_climo(climdir)
            verf.cfs_clim_grb2nc(climdir)

        # OK, we have the data downloaded, interpolated to 0.5deg latlon,
        # and converted to netcdfs (one per variable)
        # Now we just need to load them in one xarray and select the desired dates
        print('Loading CFSR climatology from netcdfs...')
        climo = xarray.open_mfdataset(climfiles, chunks=chunks)
        # Now let's only select the dates we want
        climdates = [datetime.utcfromtimestamp(dt.tolist()/1e9) for dt in climo['time'].values]
        des_dates = [(idate + timedelta(hours=6*x)).replace(year=1984) for x in \
                     range(int(timedelta_hours(idate,fdate)/6) + 1)]
        des_inds = np.where([clim_dt in des_dates for clim_dt in climdates])[0]
        # Select ONLY these dates!
        print('  selecting only desired dates...')
        climo = climo.isel(time=des_inds)
        # If our date range spans a new year, roll the data accordingly
        if fdate.year - idate.year == 1:
            # How many dates are in the new year?
            des_dates = [idate + timedelta(hours=6*x) for x in \
                         range(int(timedelta_hours(idate,fdate)/6) + 1)]
            nroll = len(np.where([dt.year==fdate.year for dt in des_dates])[0])
            climo = climo.roll(time=-nroll)

        # Rename the dimensions
        climo.rename({'latitude' : 'nLats', 'longitude' : 'nLons', 'time' : 'Time'}, inplace=True)

        # Store metadata
        climo.attrs.update(idate=idate, dt=6, type='CFSRclim', workdir=climdir)
        climo.__class__ = cls # Change the class to LatLonData
        # Add 'lat' and 'lon' variables so functions below still work
        climo.update(climo.assign(lat=climo.variables['nLats']))
        climo.update(climo.assign(lon=climo.variables['nLons']))
        # Change precipitation from mm/s to mm/h
        if 'prate1h' in climo.variables.keys():
            climo['prate1h'] *= 3600.
            climo.update(climo.assign(prate1d=climo.variables['prate1h']*24.))
        return climo

#==== Functions to get various useful attributes ==============================
    def idate(self):
        return self.attrs['idate']
    def dt(self):
        return self.attrs['dt']
    def dx(self):
        return self['lon'].values[1] - self['lon'].values[0]
    def dy(self):
        return self['lat'].values[1] - self['lat'].values[0]
    def type(self):
        return self.attrs['type']
    def workdir(self):
        return self.attrs['workdir']
    def ntimes(self):    
        return self.dims['Time']
    def ny(self):
        return self.dims['nLats']
    def nx(self):
        return self.dims['nLons']
    def vars(self):
        return [x for x in self.variables.keys() if x not in ['lat','lon']]
    def nvars(self):
        return len(self.vars())
    def vdates(self):
        return np.array([self.idate() +  timedelta(hours=t*self.dt()) for t in range(self.ntimes())])
    def leadtimes(self):
        return [timedelta_hours(self.idate(), d) for d in self.vdates()]
        
#==== Get the lat/lon grid and area weights ==================================
    def latlons(self):
        """ Returns 1D lat and lon grids """
        return self['lat'].values, self['lon'].values
    
    def area_weights(self, asdataarray=False):
        if asdataarray:
            return np.cos(np.radians(self['lat']))
        else:
            return np.cos(np.radians(self['lat'].values))
    
#==== Function to return a DataArray of the desired variable ==================
    def get_var(self, vrbl):
        if vrbl not in self.vars():
            raise ValueError('"{}" not in list of valid variables'.format(vrbl))
        return self.data_vars[vrbl]
    
#==== Drop variables from the xarray Dataset ===================================
    def dropvars(self, vrbls):
        """ Drops all variables in vrbls """
        return self.drop(vrbls)
    
    def keepvars(self, vrbls):
        """ Drops all variables NOT in vrbls """
        return self.dropvars([var for var in self.vars() if var not in vrbls])
        
#==== Compute the total precipitation rate from rainc + rainnc ================
    def compute_preciprate(self, dt=3):
        assert dt % self.dt() == 0
        assert self.type() == 'MPAS'
        raint = self['rainc'] + self['rainnc']
        if dt==24:
            prate = (raint - raint.shift(Time=1))
            prate *= 24/self.dt()
            unitstr = 'mm/d'
            varname = 'prate1d'
        else:
            prate = (raint - raint.shift(Time=int(dt/self.dt())))
            if dt==1: unitstr = 'mm/h'
            else: unitstr = 'mm/{}h'.format(int(dt))
            varname = 'prate{}h'.format(dt)
        prate = prate.assign_attrs(units=unitstr, long_name='precipitation rate')
        assignvar = {varname : prate}
        self.update(self.assign(**assignvar))
        print('Created new variable: "{}"'.format(varname))
    
#==== Transform the dataset so all lons are positive ==========================
    def restructure_lons(self, verbose=False):
        """
        Restructures the Dataset so that longitudes go from 0 to 360 rather 
        than -180 to 180. (Needed for some Basemap projections)
        """
        if (self['lon'].values>=0).all():
            if verbose: print('All longitudes are positive; no need to restructure!')
            return   
        # Find where the first non-negative longitude is
        li = 0
        while self['lon'].values[li] < 0.: li += 1
        # Append all the data associated with negative lons to the end
        self.update(self.roll(nLons=-li))
        # Change the negative lon values to positive lon values (by adding 360)
        newlonvalues = deepcopy(np.array(self['lon'].values[:]))
        newlonvalues[newlonvalues<0] = newlonvalues[newlonvalues<0] + 360.
        self['lon'].values = newlonvalues

#==== Project coordinates for plotting on a map ==============================
    def project_coordinates(self, m):
        """
        Projects the lat-lons onto a map projection
        """
        lo, la = np.meshgrid(self['lon'].values[:], self['lat'].values[:])
        return m(lo, la)
    
#==== Resample the fields temporally and returns the coarsened xarray =======
    def coarsen_temporally(self, new_dt):
        assert new_dt % self.dt() == 0
        dt_ratio = int(new_dt / self.dt())
        newMPASproc = self.isel(Time=np.arange(self.ntimes())[::dt_ratio])
        return newMPASproc
        
#==== Resamples the lat/lon grid by averaging within coarser grid boxes =====
    def coarsen_grid(self, newlats, newlons):
        """ 
        Uses xarray's powerful groupby tool to bin the lat/lon data into coarser
        lat/lon grid boxes (or "bins") and then average each bin to conservatively
        coarsen the data.
        """
        # get our lat/lon spacing and bins
        dx = newlats[1] - newlats[0]  # assuming uniform lat/lon grid
        # bin arrays are 1 element longer than lat/lon arrays, with the center
        # of each bin being the newlat/newlon value
        lat_bins = np.append(newlats-dx/2, newlats[-1]+dx/2)
        lon_bins = np.append(newlons-dx/2, newlons[-1]+dx/2)

        # average within the latitude bins
        newMPASproc = self.groupby_bins('lat', lat_bins, labels=newlats).mean(dim='nLats', keep_attrs=True)
        # trim off the extra dimension added to lon                
        newMPASproc = newMPASproc.assign(lon=newMPASproc['lon'].isel(lat_bins=0).drop('lat_bins'))
        # average within the longitude bins
        newMPASproc = newMPASproc.groupby_bins('lon', lon_bins, labels=newlons).mean(dim='nLons', keep_attrs=True)
        # trim off the extra dimension added to lat
        newMPASproc = newMPASproc.assign(lat=newMPASproc['lat'].isel(lon_bins=0).drop('lon_bins'))
        # rename/reorganize the dimensions
        newMPASproc = newMPASproc.rename({'lat_bins' : 'nLats' , 'lon_bins' : 'nLons'})
        newMPASproc = newMPASproc.transpose('Time','nLats','nLons')
        newMPASproc.__class__ = self.__class__
        return newMPASproc
         
#==== Return an interpolated field ==========================================
    def interpolate_field(self, field, newlats, newlons):
        from scipy.interpolate import interp2d
        print('interpolating {}...'.format(field))
        times = np.arange(self.ntimes())
        lats, lons = self.latlons()
        data = self[field].values
        if lats[1] < lats[0]:
            lats = lats[::-1]
            data = data[:,::-1,:]
        interpolated_data = np.zeros((len(times), len(newlats), len(newlons)))
        for t in times:
            print('time {} of {}'.format(t+1, len(times)))
            f = interp2d(lons, lats, data[t,:,:])
            interpolated_data[t,:,:] = f(newlons, newlats)
        return interpolated_data
                  
#==== Meridionally average a field ==========================================
    def hovmoller(self, field=None, lat_i=-15., lat_f=15.):
        lats = self['lat'].values
        yi = nearest_ind(lats, lat_i)
        yf = nearest_ind(lats, lat_f) + 1
        if field is None:
            latband = self.isel(nLats=range(yi,yf)) * self.area_weights(asdataarray=True)
        else:
            latband = self.isel(nLats=range(yi,yf))[field] * self.area_weights()[None, yi:yf, None]
        return latband.mean(dim='nLats', keep_attrs=True)
        
#==== Average all fields or a single field between two times ==================
    def compute_timemean(self, field=None, dt_i=None, dt_f=None):
        if dt_i is None or dt_f is None:
            if field is None:  return self.mean(dim='Time', keep_attrs=True)
            else:              return self[field].mean(dim='Time', keep_attrs=True)
        else:
            ti = nearest_ind(self.vdates(), dt_i)
            tf = nearest_ind(self.vdates(), dt_f) + 1
            if field is None:  return self.isel(Time=range(ti,tf)).mean(dim='Time', keep_attrs=True)
            else:              return self.isel(Time=range(ti,tf))[field].mean(dim='Time', keep_attrs=True)
        
#==== Average the data to a coarser timescale (e.g., daily, weekly) ===========
    def temporal_average(self, timescale):
        """ [timescale] should be in hours """
        assert timescale % self.dt() == 0
        indiv_times = []
        vdates = self.vdates()
        ntsteps = int(timescale/self.dt())
        for t in np.arange(0, self.ntimes()-1, ntsteps):
            avg_1time = self.compute_timemean(dt_i=vdates[t], 
                                              dt_f=vdates[t]+timedelta(hours=timescale-self.dt()))
            indiv_times.append(avg_1time)
        avgd_data = xarray.concat(indiv_times, dim='Time', data_vars='different')
        avgd_data.__class__ = self.__class__
        avgd_data.attrs.update(dt=timescale)
        return avgd_data
        
#==== Fetch the data from a subset of the grid ===============================
    def subset(self, field, ll=(-90, -80), ur=(90, 360), aw=False):
        lats, lons = self.latlons()
        yi = nearest_ind(lats, ll[0])
        yf = nearest_ind(lats, ur[0]) + 1
        xi = nearest_ind(lons, ll[1])
        xf = nearest_ind(lons, ur[1]) + 1
        subset = self[field].isel(nLats=range(yi,yf), nLons=range(xi,xf))
        if aw:
            return subset * self.area_weights()[None,yi:yf,None]
        else:
            return subset
        
#==== Average a field within some spacial domain =============================
    def spatial_average(self, field, lat_i=-90, lat_f=90, lon_i=-180, lon_f=360):
        """ Default: global mean """
        
        subset = self.subset(field, ll=(lat_i, lon_i), ur=(lat_f, lon_f), aw=True)
        return subset.mean(dim=('nLats','nLons'), keep_attrs=True)
    
#==== Get the timeseries of a given field at the desired lat/lon =============
    def get_timeseries(self, field, loc, verbose=False):
        """ Interpolation method = nearest """
        lat, lon = loc
        lats, lons = self.latlons()
        if lon < 0 and (lons>=0).all():
            lon += 360
        elif lon > 180 and not (lons>0).all(): 
            lon -= 360
        # Find the nearest point on the grid
        lat_ind = nearest_ind(lats, lat)
        lon_ind = nearest_ind(lons, lon)
        if verbose:
            print('Fetching data at {:.02f}N {:.02f}E'.format(lats[lat_ind], lons[lon_ind]))
        # Return the data at that point
        return self[field].isel(nLats=lat_ind, nLons=lon_ind).values
        
#==== Bandpass filter a desired field  ========================================
    def bandpass_filter(self, field, freq_i=1/2400., freq_f=1/480., 
                        wavenumbers=None, dim='Time'):
        #from scipy.fftpack import rfft, irfft, fftfreq
        from numpy.fft import rfft, irfft, fftfreq
        
        dimnum = self[field].dims.index(dim)
        
        if dim=='Time':
            ds = self.dt()
        elif dim=='nLats':
            ds = self['lat'].values[1] - self['lat'].values[0]
        elif dim=='nLons':
            ds = self['lon'].values[1] - self['lon'].values[0]
        else:
            raise ValueError('invalid dimension {}'.format(dim))
        
        signal = self[field].values
        W = fftfreq(self[field].shape[dimnum], d=ds)
        f_signal = rfft(signal, axis=dimnum)

        # If our original signal time was in seconds, this is now in Hz    
        cut_f_signal = f_signal.copy()
        if wavenumbers is not None and dim=='nLons':
            cut = np.zeros(np.shape(cut_f_signal))
            cut[:, :, wavenumbers] = 1
            cut_f_signal *= cut
        elif dimnum==0:
            print([(w**-1)/24 for w in W])
            cut_f_signal[(W < freq_i) + (W > freq_f), :, :] = 0
        elif dimnum==1:
            cut_f_signal[:, (W < freq_i) + (W > freq_f), :] = 0
        elif dimnum==2:
            cut_f_signal[:, :, (W < freq_i) + (W > freq_f)] = 0
        else:
            raise ValueError('Invalid dimenion number {}'.format(dimnum))

        #self[field].values = irfft(cut_f_signal, axis=dimnum)
        assignvar = {'{}_{}filt'.format(field, dim) : (('Time','nLats','nLons'), irfft(cut_f_signal, axis=dimnum))}
        self.update(self.assign(**assignvar))

#==== Apply a running mean to the data ========================================
    def running_mean(self, field, N, dim='Time'):
        from scipy.ndimage import uniform_filter1d
        
        dimnum = self[field].dims.index(dim)
        filt_data = uniform_filter1d(self[field].values, N, axis=dimnum, mode='nearest')
        assignvar = {'{}_{}-{}mean'.format(field, dim, N) : (('Time','nLats','nLons'), filt_data)}
        self.update(self.assign(**assignvar))
        
#==== Function to save the xarray Dataset to a netcdf file ====================
    def save_to_disk(self, filename='{}/mpas_forecast_{:%Y%m%d%H}.nc'):
        """ Dump this object to disk """
        self.to_netcdf(filename.format(self.workdir(), self.idate()))
        
        
################################################################################################
################################################################################################
# raw MPAS forecast output on Voronoi mesh
################################################################################################
################################################################################################

class MPASmeshData(xarray.Dataset):
    """Define a multivariate Dataset composed of MPAS forecast output."""
    
    # This is a list of all the variables describing the Voronoi mesh structure
    # (needed for navigation among the grid cells)
    global meshvars
    meshvars = ['latCell','lonCell','xCell','yCell','zCell','indexToCellID','latEdge','lonEdge','xEdge',
                'yEdge','zEdge','indexToEdgeID','latVertex','lonVertex','xVertex','yVertex','zVertex',
                'indexToVertexID','cellsOnEdge','nEdgesOnCell','nEdgesOnEdge','edgesOnCell','edgesOnEdge',
                'weightsOnEdge','dvEdge','dcEdge','angleEdge','areaCell','areaTriangle','cellsOnCell',
                'verticesOnCell','verticesOnEdge','edgesOnVertex','cellsOnVertex','kiteAreasOnVertex',
                'meshDensity','zgrid','fzm','fzp','zz']
    @classmethod
    def from_netcdf(cls, workdir, idate, dt, inputstream='diag',
                    meshinfofile='*init.nc', chunks=None):
        """
        Initializes a new MPASmeshData object when given
        a list of MPAS output files (netcdf)
        
        MPAS files are assumed to be raw output from the model, 
        e.g., history.*.nc or diag.*.nc files.
        
        Requires:
        workdir -----> string: path to the input/output directory
        idate -------> datetime object indicating the data start date
        dt ----------> the number of hours between each data time
        inputstream -> string: the MPAS output stream to be loaded (e.g., "diag" --> "diag.*.nc"
        meshinfofile > a filename (or wildcard) pointing to a netcdf with mesh info variables
        outputfile --> the name of the final (interpolated) netcdf file
        chunks ------> dictionary for chunking the xarray contents to dask arrays
        """
        assert isinstance(idate, datetime)
        ncfiles = '{}/{}.*.nc'.format(workdir, inputstream)
        forecast = xarray.open_mfdataset(ncfiles, concat_dim='Time', chunks=chunks)
        meshinfo = xarray.open_mfdataset('{}/{}'.format(workdir, meshinfofile))
        # Let's make sure that this MPAS output stream has cell/edge/vertex info
        for var in meshvars:
            if var not in forecast.variables.keys():
                meshvar = meshinfo[var]
                assignvar = {var : meshvar}
                forecast.update(forecast.assign(**assignvar))
        forecast.attrs.update(idate=idate, dt=dt, type='MPAS', workdir=workdir)
        forecast.__class__ = cls
                
        return forecast
        
    #==== Functions to get various useful attributes ==========================
    def idate(self):
        return self.attrs['idate']
    def dt(self):
        return self.attrs['dt']
    def type(self):
        return self.attrs['type']
    def workdir(self):
        return self.attrs['workdir']
    def ntimes(self): 
        return self.dims['Time']
    def ncells(self):
        return self.dims['nCells']
    def nedges(self):
        return self.dims['nEdges']
    def nvertices(self):
        return self.dims['nVertices']
    def vars(self):
        return self.variables.keys()
    def nvars(self):
        return len(self.vars())
    def vdates(self):
        return np.array([self.idate() +  timedelta(hours=t*self.dt()) for t in range(self.ntimes())])
    def leadtimes(self):
        return [timedelta_hours(self.idate(), d) for d in self.vdates()]
    
    #==== Drop variables from the xarray Dataset ===================================
    def dropvars(self, vrbls):
        """ Drops all variables in vrbls """
        return self.drop(vrbls)
    
    def keepvars(self, vrbls):
        """ Drops all variables NOT in vrbls """
        return self.dropvars([var for var in self.vars() if var not in vrbls])
    
    #==== Get the lat/lon locations of the cells/edges/vertices ===============
    def cell_latlons(self):
        return np.degrees(self['latCell'].values[0,:]), np.degrees(self['lonCell'].values[0,:])
    def edge_latlons(self):
        return np.degrees(self['latEdge'].values[0,:]), np.degrees(self['lonEdge'].values[0,:])
    def vertex_latlons(self):
        return np.degrees(self['latVertex'].values[0,:]), np.degrees(self['lonVertex'].values[0,:])
    
    #==== Compute the total precipitation rate from rainc + rainnc ============
    def compute_preciprate(self, dt=3):
        assert dt % self.dt() == 0
        raint = self['rainc'] + self['rainnc']
        prate = (raint - raint.shift(Time=int(dt/self.dt())))
        if dt==1: unitstr = 'mm/h'
        else: unitstr = 'mm/{}h'.format(int(dt))
        prate = prate.assign_attrs(units=unitstr, long_name='precipitation rate')
        varname = 'prate{}h'.format(dt)
        assignvar = {varname : prate}
        self.update(self.assign(**assignvar))
        print('Created new variable: "{}"'.format(varname))
    
    #==== Save the terrain elevation (on the native grid) as a variable =======
    def get_terrain(self, suffix='.init.nc'):
        # Because terrain is in the default diagnostics stream, we can get it
        # from a different file
        for file in os.listdir(self.workdir()):
            if file.endswith(suffix):
                ncfile = '{}/{}'.format(self.workdir(), file)
                break

        xry_dset = xarray.open_dataset(ncfile)
        tervar = xry_dset['ter']
        self.update(self.assign(ter=tervar))
        print('Created new variable: "ter"')
        
    #==== Compute the global (area-weighted) average of a field ===============
    def compute_globalmean(self, field, aw=True):
        # figure out if our field is on the cell grid or vertex grid
        if 'nCells' in self[field].dims:
            dimvar = 'nCells'
            areavar = 'areaCell'
        elif 'nVertices' in self[field].dims:
            dimvar = 'nVertices'
            areavar = 'areaTriangle'
        else:
            print('ERROR: Field "{}" is not on edge or vertex mesh.'.format(field))
            exit(1)
        if aw:
            # compute area weights
            R_e = 6.371e6
            A_e = 4. * np.pi * R_e**2
            wgts = np.divide(self[areavar].values, A_e)
            assert (wgts[0,:]==wgts[1,:]).all()
            # return the weighted average
            return (self[field] * wgts).sum(dim=dimvar, skipna=True, keep_attrs=True)
        else:
            # return the un-weighted average
            return self[field].mean(dim=dimvar, skipna=True, keep_attrs=True)
        
    #==== Get the timeseries of a given field at the desired lat/lon =============
    def get_timeseries(self, field, loc, verbose=False):
        """ Interpolation method = nearest """
        lat, lon = loc
        if lon < 0: lon += 360
        if 'nCells' in self[field].dims:
            lats, lons = self.cell_latlons(); dim = 'nCells'
        elif 'nEdges' in self[field].dims:
            lats, lons = self.edge_latlons(); dim = 'nEdges'
        elif 'nVertices' in self[field].dims:
            lats, lons = self.vertex_latlons(); dim = 'nVertices'
        # Find the nearest point in the mesh
        ind = (np.abs(lats-lat) + np.abs(lons-lon)).argmin()
        if verbose:
            print('Fetching data at {:.02f}N {:.02f}E'.format(lats[ind], lons[ind]))
        # Return the data at that point
        attrs = {dim : ind}
        return self[field].isel(**attrs).values
        
    #==== Interpolate a field onto a regular lat/lon grid =========================
    def interpolate_field(self, field, date, lats=np.arange(-90, 91, 1), 
                     lons=np.arange(0, 360, 1)):
        """ Adapted from mpas_contour_plot.py by Luke Madaus """
        from matplotlib.mlab import griddata
               
        # First, figure out our cell/edge/vertex coordinates
        if 'nCells' in self[field].dims:
            flats = self['latCell'].values
            flons = self['lonCell'].values
        elif 'nEdges' in self[field].dims:
            flats = self['latEdge'].values
            flons = self['lonEdge'].values
        elif 'nVertices' in self[field].dims:
            flats = self['latVertex'].values
            flons = self['lonVertex'].values
        else:
            raise ValueError("Unable to find lat/lon data for var: {}".format(field))
            exit(1)
        if 'Time' in self.dims:
            flons = flons[0,:] * (180./np.pi) # rads 2 degrees
            flats = flats[0,:] * (180./np.pi) # rads 2 degrees
            assert len(np.shape(flons))==len(np.shape(flats))==1
            
        # If we have times, find only the desired time
        if 'Time' not in self[field].dims:
            curfield = self[field].values[:]
        else:
            time = nearest_ind(self.vdates(), date)
            curfield = self[field].values[time,:]

        # We now have lats, lons and values
        # Now we just interpolate this to a regular grid
        # at the highest resolution
        dx = lons[1]-lons[0]
        print('Interpolating {} to {:.01f}-deg grid'.format(field, dx))
        gridded = griddata(flons, flats, curfield, lons, lats, interp='linear')
        return gridded

    #==== Calculate the approximate spherical grid spacing =========================
    def approx_dx(self):
        """ Uses the dcEdge variable to easily calculate the average distance
        between each cell and its neighbors """

        # Check to see if we already have grid spacing information
        if 'dx' in self.variables.keys():
            print('Grid spacing "dx" has already been computed.')
            return

        print('Calculating approximate grid spacing...')
        # We just need one time
        if 'Time' in self.dims:
            fcst = self.isel(Time=0)
        else:
            fcst = deepcopy(self)

        # Get cell navigation information
        nCells = fcst.dims['nCells']
        nEdgesOnCell = fcst['nEdgesOnCell'].values
        edgesOnCell = fcst['edgesOnCell'].values
        dcEdge = fcst['dcEdge'].values  # spherical distance between cells!

        # Empty array for approximate dx values
        dx = np.zeros(nCells)
        print("    Total num cells:", nCells)
        # Loop through all the cells
        for c in range(nCells):
            if c % 50000 == 0:
                print("        On:", c)

            # Each collection of edges has a length of maxEdges.  
            # Need to figure out how many edges are ACTUALLY on the cell, 
            # as the rest is just padded with junk data
            # These are the edge indices
            edge_inds = np.array(edgesOnCell[c,:nEdgesOnCell[c]])
            # Subtract one
            edge_inds -= 1
            # Get the distance between each cell and this cell (in km)
            dx_each_cell = [dcEdge[ei]/1000. for ei in edge_inds]
            # Assign the average distance as the grid spacing of this cell
            dx[c] = np.mean(dx_each_cell)

        # Finally, assign this approx. grid spacing as a new variable "dx"
        dxvar = xarray.DataArray(np.array(dx), dims={'nCells' : nCells})
        self.update(self.assign(dx=dxvar))
        return
    
    #==== Function to save the xarray Dataset to a netcdf file ====================
    def save_to_disk(self, filename='{}/mpas_raw_forecast_{:%Y%m%d%H}.nc'):
        """ Dump this object to disk """
        self.to_netcdf(filename.format(self.workdir(), self.idate()))
            
    
###############################################################################################
# extra utilities
###############################################################################################


def timedelta_hours(dt_i, dt_f):
    """ Find the number of hours between two dates """
    return int((dt_f-dt_i).days*24 + (dt_f-dt_i).seconds/3600)

def nearest_ind(array, value):
    return int((np.abs(array-value)).argmin())

def haversine(loc1, loc2, indegrees=True):
    """ Calculate distance between two latlon locations using Haversine formula"""
    R = 6371  # radius of Earth [km]
    lat1 = loc1[0]
    lat2 = loc2[0]
    dlon = loc2[1] - loc1[1]
    if indegrees:
        lat1 = np.radians(lat1) 
        lat2 = np.radians(lat2)
        dlon = np.radians(dlon)
    dlat = lat2 - lat1
    
    # Haversine formula:
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c
        
            