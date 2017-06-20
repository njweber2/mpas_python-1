#!/usr/bin/env python
"""
A library of functions for plotting MPAS output
Designed for use with the MPASmeshData class, which stores MPAS forecast output on its native mesh
"""

import numpy as np
import matplotlib.pyplot as plt
from color_maker.color_maker import color_map
from plotting_mpas_latlon import truncate_colormap, get_contour_levs, draw_fig_axes
from datetime import datetime
from mpasoutput import MPASmeshData

#############################################################################################################

def mpas_grid_to_patches(mpasfname='../output.nc', picklefile='mpas_paths.pckl',
                         idate=datetime(2017,4,1,0), dt=6, bmap=None):
    """ 
    Function to create a collection of patches in Basemap plotting
    coordinates that define the cells of the MPAS mesh 
    Written by Luke Madaus.
    """
    import _pickle as cPickle
    import matplotlib
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    print("Defining patch collection on MPAS grid...")

    if not isinstance(mpasfname, MPASmeshData):
        mpasfcst = MPASmeshData(mpasfname,idate,dt)
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
        if c % 50000 == 0:
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
    outfile = open(picklefile,'wb')
    cPickle.dump(p, outfile)
    outfile.close()

    return p

#############################################################################################################

def pcolor_on_mesh(m, ax, cax, fcst_xry, var='ter', picklefile=None, vmin=0., vmax=3000.,
                   cmap=color_map('OceanLakeLandSnow'), title=None, time=0):
    """
    Make a pcolor-type plot on the MPAS native mesh
    fcst_xry ---> an MPASmeshData object
    """
    import _pickle as cPickle
    
    if picklefile is None:
        picklefile = '{}/stere_conus_24k_mesh.pckl'.format(fcst_xry.workdir)
        
    # Create mesh patch file if it doesn't exist already
    try:
        patchfile = open(picklefile,'rb')
        p = cPickle.load(patchfile)
        patchfile.close()
    except:
        p = mpas_grid_to_patches(mpasfname=fcst_xry, picklefile=picklefile, bmap=m)
        
    # Get the field grid for plotting
    if 'Time' in fcst_xry[var].dims:
        field = fcst_xry[var].values[time,:]
    else:
        field = fcst_xry[var].values[:]
        
    # Plot the field
    p.set_array(field)
    # Remove lines delineating each cell
    p.set_edgecolors('none')
    p.set_antialiaseds(False)
    # Set the colormap
    p.set_cmap(cmap)
    p.set_clim(vmin=vmin, vmax=vmax)
    # Add the collection with its colorized properties to the plot
    ax.add_collection(p)
    plt.colorbar(p, cax=cax)
    if title is not None:
        ax.text(0.0, 1.015, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=15)
        ax.text(1.0, 1.009, 'valid: {:%Y-%m-%d %H:00}'.format(fcst_xry.vdates()[time]),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=12)
        ax.text(1.0, 1.048, 'init: {:%Y-%m-%d %H:00}'.format(fcst_xry.idate), 
                transform=ax.transAxes, ha='right', va='bottom', fontsize=12)
        
#############################################################################################################

def plot_mesh(m, ax, fcst_xry, picklefile=None, title='MPAS Voronoi mesh'):
    """
    Draw the MPAS Voronoi mesh over map m
    fcst_xry ---> an MPASmeshData object
    """
    import _pickle as cPickle
    
    if picklefile is None:
        picklefile = '{}/stere_conus_24k_mesh.pckl'.format(fcst_xry.workdir)
        
    # Create mesh patch file if it doesn't exist already
    try:
        patchfile = open(picklefile,'rb')
        p = cPickle.load(patchfile)
        patchfile.close()
    except:
        p = mpas_grid_to_patches(mpasfname=fcst_xry, picklefile=picklefile, bmap=m)
    
    # Fill/color and oceans and land
    m.drawmapboundary(fill_color='#99ffff')
    m.fillcontinents(color='#cc9966', lake_color='#99ffff',zorder=0)
      
    # Add the collection to the plot
    ax.add_collection(p)
    # Delineate each cell with black lines, no fill
    p.set_facecolor('none')
    p.set_edgecolors('k')
    if title is not None:
        ax.text(0.0, 1.015, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=15)
        

