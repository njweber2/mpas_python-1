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
    def from_netcdf(cls, ncfile, idate, dt):
        """
        Initializes a new MPASprocessed object when given
        a processed MPAS output file (netcdf)
        
        MPAS files are assumed to be on a lat lon grid; i.e., raw MPAS diag.nc (or any 
        other stream) files converted with the convert_mpas utility:
        https://github.com/mgduda/convert_mpas/
        """
        assert isinstance(idate, datetime)
        forecast = xarray.open_dataset(ncfile)
        forecast.__class__ = cls
        # Need to store date info, because it is not stored in the MPAS netcdf
        # metadata by default
        forecast['idate'] = idate  # initialization date
        forecast['dt'] = dt        # output frequency (days)
        return forecast

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
        return self['lat'].values[:], self['lon'].values[:]
    
#==== Function to return a DataArray of the desired variable ==================
    def get_var(self, vrbl):
        if vrbl not in self.vars():
            raise ValueError('"{}" not in list of valid variables'.format(vrbl))
        return self[vrbl]
    
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
    def from_netcdf(cls, workdir, idate, dt, dropvars=[], outputstream='diag'):
        """
        Initializes a new MPASraw object when given
        a list of MPAS output files (netcdf)
        
        MPAS files are assumed to be raw output from the model, 
        e.g., history.*.nc or diag.*.nc files.
        """
        assert isinstance(idate, datetime)
        ncfiles = '{}/{}.*.nc'.format(workdir, outputstream)
        forecast = xarray.open_mfdataset(ncfiles, drop_variables=dropvars, concat_dim='Time')
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
        return self['latCell'].values[:], self['lonCell'].values[:]
    def edge_latlons(self):
        return self['latEdge'].values[:], self['lonEdge'].values[:]
    def vertex_latlons(self):
        return self['latVertex'].values[:], self['lonVertex'].values[:]
    
    #==== Compute the total precipitation rate from rainc + rainnc ============
    def compute_preciprate(self, dt=3):
        assert dt % self.dt == 0
        raint = self['rainc'] + self['rainnc']
        prate = (raint - raint.shift(Time=int(dt/self.dt)))
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
        
    #==== Function to save the xarray Dataset to a netcdf file ====================
    def save_to_disk(self, filename='{}/mpas_raw_forecast_{:%Y%m%d%H}.nc'):
        """ Dump this object to disk """
        self.to_netcdf(filename.format(self.workdir, self.idate))
            
    
########################################################################################
# extra utilities
########################################################################################


def timedelta_hours(dt_i, dt_f):
    return (dt_f-dt_i).days*24 + (dt_f-dt_i).seconds/3600
        
            