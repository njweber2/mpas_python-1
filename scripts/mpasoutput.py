#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import xarray
from datetime import datetime, timedelta
import xarray.ufuncs as xu
from copy import deepcopy
import os

########################################################################################
# MPAS forecast on a lat-lon grid
########################################################################################

class MPASprocessed(xarray.Dataset):
    """Define a multivariate Dataset composed of MPAS forecast output."""
        
    @classmethod
    def from_netcdf(cls, ncfile, idate, dt, chunks={'Time': 10}):
        """
        Initializes a new MPASprocessed object when given
        a processed MPAS output file (netcdf)
        
        MPAS files are assumed to be on a lat lon grid; i.e., raw MPAS diag.nc (or any 
        other stream) files converted with the convert_mpas utility:
        https://github.com/mgduda/convert_mpas/
        """
        forecast = xarray.open_dataset(ncfile, chunks=chunks)
        forecast.__class__ = cls
        # Need to store date info, because it is not stored in the MPAS netcdf
        # metadata by default
        forecast['idate'] = idate  # initialization date
        forecast['dt'] = dt        # output frequency (days)
        forecast['type'] = 'MPAS'
        return forecast
    
    @classmethod
    def from_latlon_grid(cls, lats, lons, workdir, idate, dt, inputstream='diag.*.nc',
                         meshinfofile='*init.nc', outputfile='diags_interp.nc',
                         chunks={'Time': 10}):
        """
        Initializes a new MPASprocessed object when given
        a desired lat/lon grid
        
        raw MPAS diag.nc read and the data are interpolated to a regular
        lat/lon grid using the convert_mpas utility:
        https://github.com/mgduda/convert_mpas/
        
        Requires:
        lats, lons --> 1D numpy arrays of latitudes and longitudes
        workdir -----> string path to the input/output
        inputstream -> wildcard indicating all the raw MPAS netcdf files to be converted
        meshinfofile > a filename (or wildcard) pointing to a netcdf with mesh info variables
        outputfile --> the name of the final (interpolated) netcdf file
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
                target.write('startlat={:.01f}\n'.format(lats[0]))
                target.write('startlon={:.01f}\n'.format(lons[0]))
                target.write('endlat={:.01f}\n'.format(lats[-1]))
                target.write('endlon={:.01f}\n'.format(lons[-1]))

            # Now build and execute our convert_mpas command
            if os.path.isfile('{}/latlon.nc'.format(workdir)):
                Popen(['rm -f {}/latlon.nc'.format(workdir)], shell=True).wait()
            print('Interpolating {} to regularly-spaced lat-lon grid...'.format(inputstream))
            start = time.time()
            command = 'cd {}; ./convert_mpas {} {}'.format(workdir,meshinfofile,inputstream)
            Popen([command], shell=True).wait()   # will generate a latlon.nc file
            end = time.time()
            print('Elapsed time: {:.2f} min'.format((end-start)/60.))
            # rename the output file
            Popen(['cd {}; mv latlon.nc {}'.format(workdir, outputfile)], shell=True).wait()
        
        # Finally, create our MPASprocessed object like normal
        forecast = xarray.open_dataset('{}/{}'.format(workdir, outputfile), chunks=chunks)
        forecast.__class__ = cls
        # Need to store date info, because it is not stored in the MPAS netcdf
        # metadata by default
        forecast['idate'] = idate  # initialization date
        forecast['dt'] = dt        # output frequency (days)
        forecast['type'] = 'MPAS'
#        # Need to add lat/lon info
#        latvar = DataArray(lats, name='lat', dims={'nLats':len(lats)})
#        lonvar = DataArray(lons, name='lon', dims={'nLons':len(lons)})
#        forecast.update(forecast.assign(lat=latvar, lon=lonvar))
        return forecast

    @classmethod
    def from_GFS_netcdf(cls, workdir, idate, fdate, ncfile='gfs_analyses.nc',
                       chunks={'time': 10}):
        """
        Initializes a new MPASprocessed object when given
        a 3-hourly GFS analysis file (netcdf)
        """
        import os
        import verification as verf
        infile = '{}/{}'.format(workdir, ncfile)
        if not os.path.isfile(infile):
            print('File {} not found!'.format(infile))
            verf.download_gfsanl(idate, fdate, workdir)
            verf.convert_gfs_grb2nc(workdir, outfile=ncfile)
        analyses = xarray.open_dataset(infile, chunks=chunks)
        analyses.__class__ = cls
        # Need to store date info, because it is not stored in the MPAS netcdf
        # metadata by default
        analyses['idate'] = idate  # initialization date
        analyses['dt'] = 3        # output frequency (days)
        analyses['type'] = 'GFS'
        # Rename the coordinates/dims so the functions below still work
        analyses.rename({'latitude' : 'nLats', 'longitude' : 'nLons', 'time' : 'Time'}, inplace=True)
        analyses.update(analyses.assign(lat=analyses.variables['nLats']))
        analyses.update(analyses.assign(lon=analyses.variables['nLons']))
        return analyses
    
    # For adding the "idate" and "dt" items above
    def __setitem__(self, key, value):
        self.__dict__[key] = value

#==== Functions to get various useful attributes ==============================
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
        return np.array([self.idate +  timedelta(hours=t*self.dt) for t in range(self.ntimes())])
    def leadtimes(self):
        return [timedelta_hours(self.idate, d) for d in self.vdates()]
        
#==== Get the lat/lon grid ====================================================
    def latlons(self):
        """ Returns 1D lat and lon grids """
        return self['lat'].values, self['lon'].values
    
    def area_weights(self):
        return np.cos(np.radians(self['lat'].values))
    
#==== Function to return a DataArray of the desired variable ==================
    def get_var(self, vrbl):
        if vrbl not in self.vars():
            raise ValueError('"{}" not in list of valid variables'.format(vrbl))
        return self.data_vars[vrbl]
    
#==== Compute the total precipitation rate from rainc + rainnc ============
    def compute_preciprate(self, dt=3):
        assert dt % self.dt == 0
        assert self.type == 'MPAS'
        raint = self['rainc'] + self['rainnc']
        prate = (raint - raint.shift(Time=int(dt/self.dt)))
        if dt==1: unitstr = 'mm/h'
        else: unitstr = 'mm/{}h'.format(int(dt))
        prate = prate.assign_attrs(units=unitstr, long_name='precipitation rate')
        varname = 'prate{}h'.format(dt)
        assignvar = {varname : prate}
        self.update(self.assign(**assignvar))
        print('Created new variable: "{}"'.format(varname))
    
#==== Transform the dataset so all lons are positive ==========================
    def restructure_lons(self):
        """
        Restructures the Dataset so that longitudes go from 0 to 360 rather 
        than -180 to 180. (Needed for some Basemap projections)
        """
        if (self['lon'].values>=0).all():
            print('All longitudes are positive; no need to restructure!')
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
    def temporally_coarsen(self, new_dt):
        assert new_dt % self.dt == 0
        dt_ratio = int(new_dt / self.dt)
        newMPASproc = self.isel(Time=np.arange(self.ntimes())[::dt_ratio])
        newMPASproc.__setitem__('dt', new_dt)
        newMPASproc.__setitem__('idate', self.idate)
        newMPASproc.__setitem__('type', self.type)
        return newMPASproc
        
#==== Meridionally average a field ==========================================
    def hovmoller(self, field, lat_i=-15., lat_f=15.):
        lats = self['lat'].values
        yi = nearest_ind(lats, lat_i)
        yf = nearest_ind(lats, lat_f) + 1
        return self.isel(nLats=range(yi,yf))[field].mean(dim='nLats', keep_attrs=True)
        
#==== Compute the temporal average of a field ================================
    def compute_timemean(self, field, dt_i=None, dt_f=None):
        if dt_i is None or dt_f is None:
            return self[field].mean(dim='Time', keep_attrs=True)
        else:
            ti = nearest_ind(self.vdates, dt_i)
            tf = nearest_ind(self.vdates, dt_f) + 1
            return self.isel(Time=range(ti,tf))[field].mean(dim='Time', keep_attrs=True)
    
#==== Get the timeseries of a given field at the desired lat/lon =============
    def get_timeseries(self, field, loc):
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
        print('Fetching data at {:.02f}N {:.02f}E'.format(lats[lat_ind], lons[lon_ind]))
        # Return the data at that point
        return self[field].isel(nLats=lat_ind, nLons=lon_ind).values
    
#==== Function to save the xarray Dataset to a netcdf file ====================
    def save_to_disk(self, filename='mpas_forecast_{:%Y%m%d%H}.nc'):
        """ Dump this object to disk """
        self.to_netcdf(filename.format(self.idate))
        
        
        
########################################################################################
# raw MPAS forecast output on Voronoi mesh
########################################################################################


class MPASraw(xarray.Dataset):
    """Define a multivariate Dataset composed of MPAS forecast output."""
        
    @classmethod
    def from_netcdf(cls, workdir, idate, dt, dropvars=[], outputstream='diag',
                    chunks={'Time': 10}):
        """
        Initializes a new MPASraw object when given
        a list of MPAS output files (netcdf)
        
        MPAS files are assumed to be raw output from the model, 
        e.g., history.*.nc or diag.*.nc files.
        """
        assert isinstance(idate, datetime)
        ncfiles = '{}/{}.*.nc'.format(workdir, outputstream)
        forecast = xarray.open_mfdataset(ncfiles, drop_variables=dropvars, 
                                         concat_dim='Time', chunks=chunks)
        forecast.__class__ = cls
        # Let's make sure that this MPAS output stream has cell/edge/vertex info
        for var in ['cellsOnCell', 'cellsOnEdge', 'cellsOnVertex']:
            assert var in forecast.variables.keys()
        # Need to store date info, because it is not stored in the MPAS netcdf
        # metadata by default
        forecast['idate'] = idate  # initialization date
        forecast['dt'] = dt        # output frequency (days)
        # store the working directory as an object attribute (for I/O)
        forecast['workdir'] = workdir
        return forecast

    # For adding the "idate" and "dt" items above
    def __setitem__(self, key, value):
        self.__dict__[key] = value
        
    #==== Functions to get various useful attributes ==========================
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
        return np.array([self.idate +  timedelta(hours=t*self.dt) for t in range(self.ntimes())])
    def leadtimes(self):
        return [timedelta_hours(self.idate, d) for d in self.vdates()]
    
    #==== Get the lat/lon locations of the cells/edges/vertices ===============
    def cell_latlons(self):
        return np.degrees(self['latCell'].values[0,:]), np.degrees(self['lonCell'].values[0,:])
    def edge_latlons(self):
        return np.degrees(self['latEdge'].values[0,:]), np.degrees(self['lonEdge'].values[0,:])
    def vertex_latlons(self):
        return np.degrees(self['latVertex'].values[0,:]), np.degrees(self['lonVertex'].values[0,:])
    
    #==== Compute the total precipitation rate from rainc + rainnc ============
    def compute_preciprate(self, dt=3):
        assert dt % self.dt == 0
        raint = self['rainc'] + self['rainnc']
        prate = (raint - raint.shift(Time=int(dt/self.dt)))
        if dt==1: unitstr = 'mm/h'
        else: unitstr = 'mm/{}h'.format(int(dt))
        prate = prate.assign_attrs(units=unitstr, long_name='precipitation rate')
        varname = 'prate{}h'.format(dt)
        attrs = {varname : prate}
        self.update(self.assign(**attrs))
        print('Created new variable: "{}"'.format(varname))
    
    #==== Save the terrain elevation (on the native grid) as a variable =======
    def get_terrain(self, suffix='.init.nc'):
        # Because terrain is in the default diagnostics stream, we can get it
        # from a different file
        for file in os.listdir(self.workdir):
            if file.endswith(suffix):
                ncfile = '{}/{}'.format(self.workdir, file)
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
    def get_timeseries(self, field, loc):
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
        print('Fetching data at {:.02f}N {:.02f}E'.format(lats[ind], lons[ind]))
        # Return the data at that point
        attrs = {dim : ind}
        return self[field].isel(**attrs).values
        
    #==== Interpolate a field onto a regular lat/lon grid =========================
    def interp_field(self, field, date, lats=np.arange(-90, 91, 1), 
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
        if 'Time' not in self.dims:
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

    
    #==== Function to save the xarray Dataset to a netcdf file ====================
    def save_to_disk(self, filename='{}/mpas_raw_forecast_{:%Y%m%d%H}.nc'):
        """ Dump this object to disk """
        self.to_netcdf(filename.format(self.workdir, self.idate))
            
    
########################################################################################
# extra utilities
########################################################################################


def timedelta_hours(dt_i, dt_f):
    return (dt_f-dt_i).days*24 + (dt_f-dt_i).seconds/3600

def nearest_ind(array, value):
    return int((np.abs(array-value)).argmin())
        
            