#!/usr/bin/env python

"""
Script to do  pcolor-style MPAS model output using Basemap
and matplotlib.pyplot

Author: Luke Madaus

Modified by: Nick Weber (May 2017)
 - added MPASraw (xarray) functionality
 - updated for use in Python 3.x
"""
from mpasoutput import MPASraw
import numpy as np
import matplotlib
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
from datetime import datetime

def main():
    ncfiles = '/glade/scratch/njweber2/experiments/24km_channel_test_20170401/history*.nc'
    mpas_fcst = MPASraw.from_netcdf(ncfiles, datetime(2017,4,1,0), 6)
    pcolor_fill(mpas_fcst,var='t2m', export=False)

def pcolor_fill(fcst, var='t2m', picklefile='mpas_paths.pckl', plevel=None, export=False):
    """ Function to do a pcolor-mesh-type plot of "var" on the optional level "plevel"
    """
    import _pickle as cPickle
    # Check to see that this variable exists
    if var not in fcst.variables.keys():
        print("ERROR: Cannot find variable:", var)
        exit(1)


    # Set up the plot
    m = make_map(lons=None, lats=None)


    # Load the field and get its dimensions
    field = fcst.variables[var]
    field_dims = field.dims
    field_shape = field.shape

    # Main deal with MPAS data is that u,v are defined on edges of cells while
    # other variables are based on the cells.  Need to sort out what our
    # defining unit is
    if 'nEdges' in field_dims or 'nVertices' in field_dims:
        # May just convert this back to a gridded contour plot
        # Not implemented for now
        print("Edge or vertex data not implemented for pcolor-type plot")
        exit(1)
        
        
    # Going to create a collection of patches in matplotlib to enable speedy
    # plotting.  Call this function to do that if there is no archived patch
    # collection pickle file already.  It takes several minutes to generate the initial
    # patch collection (could probably be parallelized using multiprocessing or
    # optimized some other way; will leave that to future), so be warned.  I
    # like to generate the collection once and call the saved file every
    # time...makes it go a  lot faster.
    try:
        patchfile = open(picklefile,'rb')
        p = cPickle.load(patchfile)
        patchfile.close()
    except:
        p = mpas_grid_to_patches(mpasfname=fcst, picklefile=picklefile, bmap=m)


    # If we have nVertLevels, figure out which vertical level we want
    vert_lev = plevel 

    # If we have times, loop through all times
    if 'Time' not in field_dims:
        timeloop = [0]
    else:
        timeloop = range(fcst.ntimes())
    for time in timeloop:
        # Get the slice of "field" that we want
        if 'Time' in field_dims:
            try:
                curfield = field[time, :, vert_lev]
            except:
                curfield = field[time,:]
        else:
            try:
                curfield = field[:, vert_lev]
            except:
                curfield = field[:]
        # Get the current time string
        timestr = fcst['xtime'].values[time].decode("utf-8").strip()
        print(timestr)

        # Now plot
        fig = plt.figure(figsize=(15,8))
        ax = plt.gca()
        # Set the plotting array of our patch collection to be curfield
        p.set_array(curfield)
        # Set this to 'none' to not plot any of the edges
        p.set_edgecolors('none')
        # Set this to False to not delineate the cells at all (even setting
        # edgecolors to 'none' will show faint cell delineations). The False
        # setting here gives a more "blended" look.
        p.set_antialiaseds(False)
        # Set the colormap here
        p.set_cmap(matplotlib.cm.spectral)
        #p.set_clim(vmin=270, vmax=290)
        # Can normalize the colormap here if desired
        #p.set_norm()
        # Add the collection with its colorized properties to the plot
        ax.add_collection(p)
        m.drawcoastlines()
        plt.colorbar(p)
        plt.title('%s   Time: %s' % (var, timestr))
        if export:
            plt.savefig('{:s}_{:s}.png'.format(var,timestr), bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            plt.close()



def mpas_grid_to_patches(mpasfname='../output.nc', picklefile='mpas_paths.pckl',
                         idate=datetime(2017,4,1,0), dt=6, bmap=None):
    """ Function to create a collection of patches in Basemap plotting
    coordinates that define the cells of the MPAS mesh """
    print("Defining patch collection on MPAS grid...")

    if not isinstance(mpasfname, MPASraw):
        mpasfcst = MPASraw(mpasfname,idate,dt)
    else:
        mpasfcst = mpasfname
        
    # We just need one time
    if 'Time' in mpasfcst.dims:
        mpasfcst = mpasfcst.isel(Time=0)

    # Get the number of cells
    nCells = mpasfcst.dims['nCells']
    nEdgesOnCell = mpasfcst['nEdgesOnCell'].values
    verticesOnCell = mpasfcst['verticesOnCell'].values
    latvertex = mpasfcst['latVertex'].values
    lonvertex = mpasfcst['lonVertex'].values

    patches = [None] * nCells
    print("    Total num cells:", nCells)
    # Need a collection of lats and lons for each vertex of each cell
    for c in range(nCells):
        if c % 15000 == 0:
            print("        On:", c)
        # Each collection of vertices has a length of maxEdges.  Need to figure
        # out how many vertices are ACTUALLY on the cell, as the rest is just
        # padded with junk data
        # These are the vertex indices
        cell_verts = verticesOnCell[c,:nEdgesOnCell[c]]
        # Add on the final point to close
        cell_verts = np.append(cell_verts,cell_verts[0:1])
        # Subtract one
        cell_verts -= 1
        # Get the latitudes and longitudes of these and convert to degrees
        vert_lats = np.array([latvertex[d] * 180./np.pi for d in cell_verts])
        vert_lons = np.array([lonvertex[d] * 180./np.pi for d in cell_verts])
        # Check for overlap of date line
        diff_lon = np.subtract(vert_lons, vert_lons[0])
        vert_lons[diff_lon > 180.0] = vert_lons[diff_lon > 180.0] - 360.0
        vert_lons[diff_lon < -180.0] = vert_lons[diff_lon < -180.0] + 360.0
        # Convert to projected coordinates
        vert_x, vert_y = bmap(vert_lons, vert_lats)
        coords = np.vstack((vert_x, vert_y))
        # Now create a path for this
        # Codes follow same format
        cell_codes = np.ones(cell_verts.shape) * mpath.Path.LINETO
        cell_codes[0] = mpath.Path.MOVETO
        cell_codes[-1] = mpath.Path.CLOSEPOLY
        cell_path = mpath.Path(coords.T, codes=cell_codes, closed=True, readonly=True)
        # Convert to a Patch and add to the list
        patches[c] = mpatches.PathPatch(cell_path)

    # Now crate a patch collection
    p = matplotlib.collections.PatchCollection(patches)
    # Archive for future use
    print("    Archiving paths...")
    import _pickle as cPickle
    outfile = open(picklefile,'wb')
    cPickle.dump(p, outfile)
    outfile.close()

    return p




def make_map(lons, lats, globalmap=False):
    """ Create a basemap object and projected coordinates for a global map """
    # MPAS cell lats and lons are projected using a cylindrical projection;
    # other projections are not recommended...
    if globalmap:
        m = Basemap(projection='cyl',llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180,
                   urcrnrlon=180, resolution='c')
    else:
        latc = centerloc[0]
        lonc = centerloc[1]
        m = Basemap(ax=ax,projection='cyl', llcrnrlat=22.,urcrnrlat=53.,\
                    llcrnrlon=-125.,urcrnrlon=-67,resolution='c')

    if lons is None or lats is None:
        # Just return the map
        return m
    else:
        xgrid, ygrid = m(lons,lats)
        return m, xgrid, ygrid
    





if __name__ == '__main__':
    main()

