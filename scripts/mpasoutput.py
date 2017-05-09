#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import xarray
from datetime import datetime, timedelta
import xarray.ufuncs as xu
from copy import deepcopy


class MPASforecast(xarray.Dataset):
    """Define a multivariate Dataset composed of MPAS forecast output."""
        
    @classmethod
    def from_netcdf(cls, ncfile, idate, dt):
        """
        Initializes a new MPASforecast object when given
        a list of MPAS output files (netcdf)
        
        MPAS files are assumed to be on a lat lon grid; i.e., raw MPAS diag.nc (or any 
        other stream) files converted with the convert_mpas utility.
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

#==== Functions to get various useful attributes of the state =================
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
        return [(d - self.idate).seconds/3600 for d in self.vdates()]
        
#==== Get the lat/lon grids for a given variable ==============================
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
