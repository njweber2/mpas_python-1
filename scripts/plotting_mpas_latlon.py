#!/usr/bin/env python
"""
A library of functions for plotting MPAS output
Designed for use with the MPASprocessed class, which stores MPAS forecast output a lat-lon grid
"""

import numpy as np
import matplotlib.pyplot as plt
from color_maker.color_maker import color_map

#############################################################################################################

def magnitude(x):
    import math
    return int(math.log10(x))

def smartround(x):
    if x<1.:
        return np.round(x, magnitude(abs(x))+1)
    else:
        return float(int(x))

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Returns a truncated version of the desired colormap.
    """
    import matplotlib.colors as colors
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
    

#############################################################################################################

def get_contour_levs(field, nlevs=8, zerocenter=False, maxperc=95):
    """
    Computes/returns evenly-spaced integer contour values for a given field, 
    wherein the maximum contour value is the [maxperc]th percentile.
    """
    perc_lo = np.percentile(field.flatten(), 100-maxperc)
    perc_hi = np.percentile(field.flatten(), maxperc)
    if zerocenter:
        maxperc = max(np.abs([perc_lo, perc_hi]))  #furthest from zero
        if (nlevs % 2 == 0): #even 
            cint = smartround(maxperc/((nlevs/2.)+1))
            half = np.array([cint + i*cint for i in range(int(nlevs/2))])
            clev = np.append(-1*half[::-1], half)
        else: #odd
            cint = smartround(maxperc/((nlevs+1.)/2.))
            half = np.array([0 + i*cint for i in range(int((nlevs+1)/2))])
            clev = np.append(-1*half[1:][::-1], half)
    else:
        cint = smartround((perc_hi-perc_lo)/nlevs)
        clev = [smartround(perc_lo)+i*cint for i in range(nlevs)]
    assert len(clev)==nlevs
    return clev

#############################################################################################################

def draw_fig_axes(proj='orthoNP', mapcol='k', figsize=(12,10), twocb=False):
    """
    Prepares a single-panel figure and map projection for plotting.
    
    Requires:
    proj -----> map projection that can be found in my_map_projections.py
    mapcol ---> color of continent boundaries
    figsize --> size (w,h) of figure
    
    Returns:
    figure, axis, colorbar axis, and Basemap objects
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import my_map_projections as mymaps

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig.subplots_adjust(top=0.92)
    m = mymaps.draw_projection(ax, proj, mapcol=mapcol)
    divider = make_axes_locatable(ax)
    if twocb:
        cax = divider.append_axes('right', size='3%', pad=0.4)
        cax2 = divider.append_axes('right', size='3%', pad=0.8)
        return fig, ax, cax, cax2, m
    else:
        cax = divider.append_axes('right', size='5%', pad=0.6)
        return fig, ax, cax, m

#############################################################################################################

def plot_vort_hgt(m, ax, cax, fcst_xry, plev, vlevs=np.arange(-0.4, 2.5, 0.4),
                  hgtintvl=60, cmap=color_map('ncar_temp'), 
                  idate=None, vdate=None, units=None, cbar=True):
    """
    Plots relative vorticity, geopotential height, and wind barbs at a given pressure level.
    
    Requires:
    m --------> a Basemap object with the desired projection
    ax -------> the axis object corresponding to m
    cax ------> an axis object for the vertically-oriented colorbar
    fcst_xry -> an MPASforecast object containing all the forecast variables
    plev -----> the pressure level (integer)
    vlevs ----> contour fill levels for plotting vorticity
    hgtintvl -> contour interval for plotting geopotential heights
    cmap -----> colormap for vorticity
    idate ----> forecast initialization date (datetime object)
    vdate ----> forecast valid date (datetime object)
    units ----> a dictionary of units for the plotted variables
    cbar -----> plot the colorbar?
    """
    # MPAS variable names
    vortvar = 'vorticity_{}hPa'.format(int(plev))
    hgtvar = 'height_{}hPa'.format(int(plev))
    uvar = 'uzonal_{}hPa'.format(int(plev))
    vvar = 'umeridional_{}hPa'.format(int(plev))
    # Select a time if fcst_xry contains multiple valid times
    if 'Time' in fcst_xry.dims:
        fcst_xry = fcst_xry.isel(Time=np.where(fcst_xry.vdates()==vdate)[0][0])
    # mapped lat/lon locations
    lon = fcst_xry['lon'].values; lat = fcst_xry['lat'].values
    x, y = fcst_xry.project_coordinates(m)
    # contour fill the vorticity
    csf = m.contourf(x, y, fcst_xry[vortvar].values, levels=vlevs, 
                    cmap=cmap, extend='both')
    if cbar: plt.colorbar(csf, cax=cax)
    # contour the geopotential height
    hlevs=np.arange(4800, 6001, hgtintvl)
    cs = m.contour(x, y, fcst_xry[hgtvar].values, levels=hlevs, linewidths=2, 
                   colors='silver')
    plt.clabel(cs, np.array(cs.levels[::2]).astype(int), fmt='%03d', inline=1, fontsize=10)
    # plot the wind barbs
    uproj, vproj, xx, yy = \
            m.transform_vector(fcst_xry[uvar].values, fcst_xry[vvar].values, 
            lon, lat, 15, 15, returnxy=True, masked=True)
    m.barbs(xx,yy,uproj,vproj,length=6,barbcolor='w',flagcolor='w',linewidth=0.5)
    # Set titles
    returns = [csf, cs]
    if idate is not None and vdate is not None and units is not None:
        maintitle = '{}hPa vorticity [{}], height [{}], and winds [{}]'
        ax.text(0.0, 1.015, maintitle.format(plev, units[vortvar],units[hgtvar], units[uvar]), 
                transform=ax.transAxes, ha='left', va='bottom', fontsize=14)
        txt = ax.text(1.0, 1.01, 'valid: {:%Y-%m-%d %H:00}'.format(vdate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
        ax.text(1.0, 1.045, 'init: {:%Y-%m-%d %H:00}'.format(idate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
        returns.append(txt)
    return returns

#############################################################################################################
        
def plot_t2m_mslp(m, ax, cax, fcst_xry, tlevs=np.arange(-40, 51, 2),
                  presintvl=4, cmap=color_map('ncar_temp'), 
                  idate=None, vdate=None, units=None, cbar=True):
    """
    Plots 2m temperature, mean sea level pressure, and 10-meter wind barbs.
    
    Requires:
    m --------> a Basemap object with the desired projection
    ax -------> the axis object corresponding to m
    cax ------> an axis object for the vertically-oriented colorbar
    fcst_xry -> an MPASforecast object containing all the forecast variables
    tlevs ----> contour fill levels for plotting 2m temperatures
    presintvl > contour interval for plotting mean sea level pressure
    cmap -----> colormap for temperature
    idate ----> forecast initialization date (datetime object)
    vdate ----> forecast valid date (datetime object)
    units ----> a dictionary of units for the plotted variables
    cbar -----> plot the colorbar?
    """
    # MPAS variable names
    tempvar = 't2m'
    mslpvar = 'mslp'
    uvar = 'u10'
    vvar = 'v10'
    # Select a time if fcst_xry contains multiple valid times
    if 'Time' in fcst_xry.dims:
        fcst_xry = fcst_xry.isel(Time=np.where(fcst_xry.vdates()==vdate)[0][0])
    # mapped lat/lon locations
    lon = fcst_xry['lon'].values; lat = fcst_xry['lat'].values
    x, y = fcst_xry.project_coordinates(m)
    # contour fill the temps
    csf = m.contourf(x, y, fcst_xry[tempvar].values, levels=tlevs, 
                    cmap=cmap, extend='both')
    if cbar: plt.colorbar(csf, cax=cax)
    # contour the geopotential height
    plevs=np.arange(940, 1041, presintvl)
    cs = m.contour(x, y, fcst_xry[mslpvar].values, levels=plevs, linewidths=2, 
                   colors='dimgrey')
    plt.clabel(cs, np.array(cs.levels[::2]).astype(int), fmt='%03d', inline=1, fontsize=10)
    # plot the wind barbs
    uproj, vproj, xx, yy = \
            m.transform_vector(fcst_xry[uvar].values, fcst_xry[vvar].values, 
            lon, lat, 15, 15, returnxy=True, masked=True)
    m.barbs(xx,yy,uproj,vproj,length=6,barbcolor='k',flagcolor='k',linewidth=0.5)
    # Set titles
    returns = [csf, cs]
    if idate is not None and vdate is not None and units is not None:
        maintitle = '2-m temperature [{}], MSLP [{}], and 10-m winds [{}]'
        ax.text(0.0, 1.015, maintitle.format(units[tempvar],units[mslpvar], units[uvar]), 
                transform=ax.transAxes, ha='left', va='bottom', fontsize=14)
        txt = ax.text(1.0, 1.01, 'valid: {:%Y-%m-%d %H:00}'.format(vdate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
        ax.text(1.0, 1.045, 'init: {:%Y-%m-%d %H:00}'.format(idate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
        returns.append(txt)
    return returns

#############################################################################################################
        
def plot_trop_precip(m, ax, cax, cax2, fcst_xry, olevs=np.arange(100, 241, 20),
                     plevs=[5,7.5,10,12.5,15,17.5,20,22.5,25,30,35,40,50,70,100,150,200,250],
                     ocmap=truncate_colormap(plt.cm.gray_r, 0.0, 0.6), 
                     pcmap=truncate_colormap(cm.s3pcpn, 0.01, 1.0),
                     idate=None, vdate=None, units=None, cbar=True):
    """
    Plots TOA OLR, precipitation, and 850 hPa wind barbs.
    
    Requires:
    m --------> a Basemap object with the desired projection
    ax -------> the axis object corresponding to m
    cax ------> an axis object for the vertically-oriented colorbar
    fcst_xry -> an MPASforecast object containing all the forecast variables
    olevs ----> contour fill levels for plotting OLR
    precintvl > contour interval for plotting precipitation
    ocmap ----> colormap for OLR
    pcmap ----> colormap for preicpitation
    idate ----> forecast initialization date (datetime object)
    vdate ----> forecast valid date (datetime object)
    units ----> a dictionary of units for the plotted variables
    cbar -----> plot the colorbar?
    """
    # MPAS variable names
    olrvar = 'olrtoa'
    precvar = 'rainc'
    uvar = 'uzonal_850hPa'
    vvar = 'umeridional_850hPa'
    # colorfill land and sea
    m.drawmapboundary(fill_color='#99ffff')
    m.fillcontinents(color='#cc9966', lake_color='#99ffff', zorder=0)
    # Select a time if fcst_xry contains multiple valid times
    if 'Time' in fcst_xry.dims:
        fcst_xry = fcst_xry.isel(Time=np.where(fcst_xry.vdates()==vdate)[0][0])
    # mapped lat/lon locations
    lon = fcst_xry['lon'].values; lat = fcst_xry['lat'].values
    x, y = fcst_xry.project_coordinates(m)
    # contour fill the olr
    csf = m.contourf(x, y, fcst_xry[olrvar].values, levels=olevs, 
                    cmap=ocmap, extend='min')
    if cbar: plt.colorbar(csf, cax=cax)
    # contour the precipitation
    csf2 = m.contourf(x, y, fcst_xry[precvar].values, levels=plevs, 
                      cmap=pcmap, extend='max')
    if cbar: plt.colorbar(csf2, cax=cax2)
    # plot the wind barbs
    uproj, vproj, xx, yy = \
            m.transform_vector(fcst_xry[uvar].values, fcst_xry[vvar].values, 
            lon, lat, 15, 15, returnxy=True, masked=True)
    m.barbs(xx,yy,uproj,vproj,length=6,barbcolor='k',flagcolor='k',linewidth=0.5)
    # Set titles
    returns = [csf, csf2]
    if idate is not None and vdate is not None and units is not None:
        maintitle = 'OLR [{}], precipitation [{}], and 850-hPa winds [{}]'
        ax.text(0.0, 1.015, maintitle.format(units[olrvar],units[precvar], units[uvar]), 
                transform=ax.transAxes, ha='left', va='bottom', fontsize=14)
        txt = ax.text(1.0, 1.01, 'valid: {:%Y-%m-%d %H:00}'.format(vdate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
        ax.text(1.0, 1.045, 'init: {:%Y-%m-%d %H:00}'.format(idate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
        returns.append(txt)
    return returns

