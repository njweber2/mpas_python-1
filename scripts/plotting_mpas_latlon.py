#!/usr/bin/env python
"""
A library of functions for plotting MPAS output
Designed for use with the LatLonData class, which stores MPAS forecast output a lat-lon grid
"""

import numpy as np
import matplotlib.pyplot as plt
from color_maker.color_maker import color_map
from color_maker.nonlinear_cmap import nlcmap
from copy import deepcopy

##### Some functions for alterind colormaps ##################################################################

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
    
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    try:
        base = color_map(base_cmap)
    except:
        base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

#############################################################################################################

def get_contour_levs(field, nlevs=8, zerocenter=False, maxperc=99):
    """
    Computes/returns evenly-spaced integer contour values for a given field, 
    wherein the maximum contour value is the [maxperc]th percentile.
    
    Requires:
    field ------> multidimensional array of data
    nlevs ------> number of contour levels (int)
    zerocenter -> do we want zero to be the at the center? (i.e., diverging colormap?)
    maxperc ----> data percentile to use for max/min of the contour levels
    
    Returns:
    clev -------> list of contour levels (float)
    """
    # Find the upper and lower percentiles
    perc_lo = np.nanpercentile(field.flatten(), 100-maxperc)
    perc_hi = np.nanpercentile(field.flatten(), maxperc)
    if zerocenter:
        # Create diverging contour levels
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
        # Create uniformly increasing levels from the lower to upper percentile
        cint = smartround((perc_hi-perc_lo)/nlevs)
        clev = [smartround(perc_lo)+i*cint for i in range(nlevs)]
    assert len(clev)==nlevs
    return clev

#############################################################################################################

def draw_fig_axes(proj='orthoNP', mapcol='k', figsize=(12,10), nocb=False):
    """
    Prepares a single-panel figure and map projection for plotting.
    
    Requires:
    proj -----> map projection that can be found in my_map_projections.py
    mapcol ---> color of continent boundaries
    figsize --> size (w,h) of figure
    
    Returns:
    fig, ax, cax, m ----> figure, axis, colorbar axis, and Basemap objects
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import map_projections as mymaps

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig.subplots_adjust(top=0.92)
    m = mymaps.draw_projection(ax, proj, mapcol=mapcol)
    
    if nocb:
        return fig, ax, m
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.6)
        return fig, ax, cax, m


#############################################################################################################
        
def simple_contourf(m, ax, cax, x, y, field, levs=None, cmap=color_map('ncar_temp'), varname=None,
                    idate=None, vdate=None, units=None, cbar=True, div=False):
    """
    Takes a 2D field and produces a contourf plot on a map.
    
    Requires:
    m --------> a Basemap object with the desired projection
    ax -------> the axis object corresponding to m
    cax ------> an axis object for the vertically-oriented colorbar
    x, y -----> lat/lon coordinates projected onto Basemap m
    field ----> a 2D numpy array
    cmap -----> colormap
    varname --> the name of the plotted variable
    idate ----> forecast initialization date (datetime object)
    vdate ----> forecast valid date (datetime object)
    units ----> a string with the units of the field
    cbar -----> plot the colorbar?
    div ------> is this a diverging colormap? (neg/pos values centered at 0)
    
    Returns:
    csf ------> the contour fill object
    txt ------> (optionally) the title text object
    """
    assert len(np.shape(field))==2
    # get the contourf levels
    if levs is None:
        levs = get_contour_levs(field, nlevs=8, zerocenter=div)
    
    # contour fill
    csf = m.contourf(x, y, field, levels=levs, cmap=cmap, extend='both')
    if cbar: plt.colorbar(csf, cax=cax)
    # Set titles
    returns = [csf]
    if units is not None:
        maintitle = '{} [{}]'.format(varname, units)
        ax.text(0.0, 1.015, maintitle, transform=ax.transAxes, ha='left', va='bottom', fontsize=12)
    if idate is not None and vdate is not None:
        txt = ax.text(1.0, 1.01, 'valid: {:%Y-%m-%d %H:00}'.format(vdate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=10)
        ax.text(1.0, 1.05, 'init: {:%Y-%m-%d %H:00}'.format(idate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=10)
        returns.append(txt)
    return returns    
    #############################################################################################################

def plot_vort_hgt(m, ax, cax, fcst_xry, plev, vlevs=np.arange(-0.2, 5.3, 0.1),
                  hgtintvl=60, cmap=color_map('ncar_temp'), 
                  idate=None, vdate=None, cbar=True, swaplons=False):
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
    cbar -----> plot the colorbar?
    swaplons -> reorder data so longitudes go from 0 to 360?
    
    Returns:
    csf ------> the contour fill object
    cs -------> the contour object
    txt ------> (optionally) the title text object
    """
    # MPAS variable names
    vortvar = 'vorticity_{}hPa'.format(int(plev))
    hgtvar = 'height_{}hPa'.format(int(plev))
    uvar = 'uzonal_{}hPa'.format(int(plev))
    vvar = 'umeridional_{}hPa'.format(int(plev))
    # If the projection (e.g., Mercator) crosses 180, we need to restructure the longitudes
    fcst = deepcopy(fcst_xry)
    if swaplons: 
        fcst.restructure_lons()
    # Select a time if fcst_xry contains multiple valid times
    if 'Time' in fcst.dims:
        fcst = fcst.isel(Time=np.where(fcst_xry.vdates()==vdate)[0][0])
    # mapped lat/lon locations
    lon = fcst['lon'].values; lat = fcst['lat'].values
    x, y = fcst.project_coordinates(m)
    # contour fill the absolute vorticity
    f = 2 *  7.2921e-5 * np.sin(np.radians(lat))[:,None]  # planetary vorticity
    csf = m.contourf(x, y, (fcst[vortvar].values + f) *10**4, levels=vlevs, 
                    cmap=cmap, extend='both')
    if cbar: plt.colorbar(csf, cax=cax)
    # contour the geopotential height
    hlevs=np.arange(4800, 6001, hgtintvl)
    cs = m.contour(x, y, fcst[hgtvar].values, levels=hlevs, linewidths=2, 
                   colors='silver')
    plt.clabel(cs, np.array(cs.levels[::2]).astype(int), fmt='%03d', inline=1, fontsize=10)
    # plot the wind barbs
    uproj, vproj, xx, yy = \
            m.transform_vector(fcst[uvar].values, fcst[vvar].values, 
            lon, lat, 15, 15, returnxy=True, masked=True)
    m.barbs(xx,yy,uproj,vproj,length=6,barbcolor='w',flagcolor='w',linewidth=0.5)
    # Set titles
    returns = [csf, cs]
    if idate is not None and vdate is not None:
        maintitle = str(plev)+'-hPa absolute vorticity [10$^{-4}$ s$^{-1}$], height [m], and winds [m/s]'
        ax.text(0.0, 1.015, maintitle, transform=ax.transAxes, ha='left', va='bottom', fontsize=13)
        txt = ax.text(1.0, 1.01, 'valid: {:%Y-%m-%d %H:00}'.format(vdate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
        ax.text(1.0, 1.045, 'init: {:%Y-%m-%d %H:00}'.format(idate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
        returns.append(txt)
    return returns

#############################################################################################################
        
def plot_t2m_mslp(m, ax, cax, fcst_xry, tlevs=np.arange(-20, 41, 2),
                  presintvl=4, cmap=color_map('ncar_temp'), 
                  idate=None, vdate=None, cbar=True, swaplons=False):
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
    cbar -----> plot the colorbar?
    swaplons -> reorder data so longitudes go from 0 to 360?
    
    Returns:
    csf ------> the contour fill object
    cs -------> the contour object
    txt ------> (optionally) the title text object
    """
    # MPAS variable names
    tempvar = 't2m'
    mslpvar = 'mslp'
    uvar = 'u10'
    vvar = 'v10'
    # If the projection (e.g., Mercator) crosses 180, we need to restructure the longitudes
    fcst = deepcopy(fcst_xry)
    if swaplons: 
        fcst.restructure_lons()
    # Select a time if fcst_xry contains multiple valid times
    if 'Time' in fcst.dims:
        fcst = fcst.isel(Time=np.where(fcst_xry.vdates()==vdate)[0][0])
    # mapped lat/lon locations
    lon = fcst['lon'].values; lat = fcst['lat'].values
    x, y = fcst.project_coordinates(m)
    # contour fill the temps
    t2m = fcst[tempvar].values - 273.15 # K to C
    csf = m.contourf(x, y, t2m, cmap=cmap, extend='both', levels=tlevs)
    if cbar: plt.colorbar(csf, cax=cax)
    # contour the mslp
    plevs=np.arange(920, 1041, presintvl)
    mslp = fcst[mslpvar].values/100. # Pa to hPa
    cs = m.contour(x, y, mslp, levels=plevs, linewidths=2, colors='dimgrey')
    plt.clabel(cs, np.array(cs.levels[::2]).astype(int), fmt='%03d', inline=1, fontsize=10)
    # plot the wind barbs
    u = fcst[uvar].values * 1.94384 # m/s to kts
    v = fcst[vvar].values * 1.94384 # m/s to kts
    uproj, vproj, xx, yy = m.transform_vector(u, v, lon, lat, 15, 15, returnxy=True, masked=True)
    m.barbs(xx,yy,uproj,vproj,length=6,barbcolor='k',flagcolor='k',linewidth=0.5)
    # Set titles
    returns = [csf, cs]
    if idate is not None and vdate is not None:
        maintitle = '2-m temperature [C], MSLP [hPa], and 10-m winds [kts]'
        ax.text(0.0, 1.015, maintitle, transform=ax.transAxes, ha='left', va='bottom', fontsize=14)
        txt = ax.text(1.0, 1.01, 'valid: {:%Y-%m-%d %H:00}'.format(vdate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
        ax.text(1.0, 1.045, 'init: {:%Y-%m-%d %H:00}'.format(idate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
        returns.append(txt)
    return returns

#############################################################################################################
        
def plot_precip_mslp(m, ax, cax, fcst_xry, plevs=[.005,.01,.02,.05,.1,.25,.5,.75,1.,1.5,2.,3.],
                     presintvl=4, cmap=color_map('ncar_precip'), pdt=3,
                     idate=None, vdate=None, cbar=True, swaplons=False):
    """
    Plots 2m temperature, mean sea level pressure, and 10-meter wind barbs.
    
    Requires:
    m --------> a Basemap object with the desired projection
    ax -------> the axis object corresponding to m
    cax ------> an axis object for the vertically-oriented colorbar
    fcst_xry -> an MPASforecast object containing all the forecast variables
    plevs ----> contour fill levels for plotting precip rate
    presintvl > contour interval for plotting mean sea level pressure
    cmap -----> colormap for temperature
    pdt ------> time interval (hours) over which to calculate the precip rate
    idate ----> forecast initialization date (datetime object)
    vdate ----> forecast valid date (datetime object)
    cbar -----> plot the colorbar?
    swaplons -> reorder data so longitudes go from 0 to 360?
    
    Returns:
    csf ------> the contour fill object
    cs -------> the contour object
    txt ------> (optionally) the title text object
    """
    cmap = nlcmap(cmap, plevs)
    # MPAS variable names
    precipvar = 'prate{}h'.format(pdt)
    mslpvar = 'mslp'
    # Compute the precip rate if it's not in our xarray
    if precipvar not in fcst_xry.variables.keys():
        fcst_xry.compute_preciprate(dt=pdt)
    # If the projection (e.g., Mercator) crosses 180, we need to restructure the longitudes
    fcst = deepcopy(fcst_xry)
    if swaplons: 
        fcst.restructure_lons()
    # Select a time if fcst_xry contains multiple valid times
    if 'Time' in fcst.dims:
        fcst = fcst.isel(Time=np.where(fcst_xry.vdates()==vdate)[0][0])
    # mapped lat/lon locations
    x, y = fcst.project_coordinates(m)
    # contour fill the precip
    precip = fcst[precipvar].values * 0.0393701  # mm to inches
    csf = m.contourf(x, y, precip, cmap=cmap, levels=plevs)
    if cbar: plt.colorbar(csf, cax=cax, ticks=plevs)
    # contour the mslp
    plevs=np.arange(920, 1041, presintvl)
    mslp = fcst[mslpvar].values/100. # Pa to hPa
    cs = m.contour(x, y, mslp, levels=plevs, linewidths=1.5, colors='k')
    plt.clabel(cs, np.array(cs.levels[::2]).astype(int), fmt='%03d', inline=1, fontsize=10)
    # Set titles
    returns = [csf, cs]
    if idate is not None and vdate is not None:
        maintitle = '{}-h precipitation [in] and MSLP [hPa]'.format(pdt)
        ax.text(0.0, 1.015, maintitle, transform=ax.transAxes, ha='left', va='bottom', fontsize=14)
        txt = ax.text(1.0, 1.01, 'valid: {:%Y-%m-%d %H:00}'.format(vdate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
        ax.text(1.0, 1.045, 'init: {:%Y-%m-%d %H:00}'.format(idate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
        returns.append(txt)
    return returns

#############################################################################################################
        
def plot_brightness_temp(m, ax, cax, fcst_xry, blevs=np.arange(-80, 41, 4),
                         bcmap=color_map('ncar_ir'), idate=None, vdate=None, 
                         cbar=True, swaplons=False):
    """
    Plots TOA OLR, precipitation, and 850 hPa wind barbs.
    
    Requires:
    m --------> a Basemap object with the desired projection
    ax -------> the axis object corresponding to m
    cax ------> an axis object for the vertically-oriented colorbar
    fcst_xry -> an MPASforecast object containing all the forecast variables
    blevs ----> contour fill levels for plotting brightness temp
    bcmap ----> colormap for brightness temp
    idate ----> forecast initialization date (datetime object)
    vdate ----> forecast valid date (datetime object)
    cbar -----> plot the colorbar?
    swaplons -> reorder data so longitudes go from 0 to 360?
    
    Returns:
    csf ------> the contour fill object
    txt ------> (optionally) the title text object
    """
    # MPAS variable names
    olrvar = 'olrtoa'
    
    # If the projection (e.g., Mercator) crosses 180, we need to restructure the longitudes
    fcst = deepcopy(fcst_xry)
    if swaplons: 
        fcst.restructure_lons()
    # Select a time if fcst_xry contains multiple valid times
    if 'Time' in fcst.dims:
        fcst = fcst.isel(Time=np.where(fcst_xry.vdates()==vdate)[0][0])
    # mapped lat/lon locations
    x, y = fcst.project_coordinates(m)
    # calculate and contour fill the brightness temperature
    brightemp = (fcst[olrvar].values / 5.67e-8)**(1/4) - 273.15
    csf = m.contourf(x, y, brightemp, levels=blevs, 
                    cmap=bcmap, extend='both')
    if cbar: plt.colorbar(csf, cax=cax)
    # Set titles
    returns = [csf]
    if idate is not None and vdate is not None:
        maintitle = 'IR brightness temperature [C]'
        ax.text(0.0, 1.015, maintitle, transform=ax.transAxes, ha='left', va='bottom', fontsize=14)
        txt = ax.text(1.0, 1.01, 'valid: {:%Y-%m-%d %H:00}'.format(vdate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
        ax.text(1.0, 1.045, 'init: {:%Y-%m-%d %H:00}'.format(idate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
        returns.append(txt)
    return returns

#############################################################################################################

def plot_hovmoller(ax, cax, field, fcst_xry, slat, nlat, xlims=None, climo=None, roll=0,
                   levs=[.25, 1., 2.5, 5., 10., 15., 20., 30., 40., 50., 75., 100., 125., 150.],
                   cmap=color_map('ncar_precip'), idate=None, units=None, cbar=True, ext='both', 
                   show_ylabels=True, title=None):
    """
    Creates a 2D (longitude-time) plot of meridionally averaged data
    
    Requires:
    ax -------> the axis object corresponding to m
    cax ------> an axis object for the vertically-oriented colorbar
    field ----> the name of the variable to be plotted (string)
    fcst_xry -> an MPASforecast object containing all the forecast variables
    slat -----> the southernmost latitude of the data to be averaged
    nlat -----> the northernmost latitude of the data to be averaged
    xlims ----> the x (longitude) limits for the plot
    climo ----> the LatLonData climatology dataset (used to calculated anomalies)
    roll -----> how many indices to roll the data in the x direction
    levs -----> contour fill levels
    cmap -----> color map for contour plot
    idate ----> forecast initialization date (datetime object)
    units ----> the units for [field]  (string)
    cbar -----> plot the colorbar?
    ext ------> 'extend' keyword for contourf function
    show_ylabels -> show ytick labels?
    title ----> the title of the figure (string)
    
    Returns:
    csf ------> the contour fill object
    hov4plot -> the meridionally averaged data
    """
    import matplotlib.dates as mdates
    
    if roll != 0: assert xlims is None
    fcst_xry.restructure_lons()
    # average the field from lat_i to lat_f
    hov = fcst_xry.hovmoller(field, lat_i=slat, lat_f=nlat).values
    if climo is not None:
        hov -= climo.hovmoller(field, lat_i=slat, lat_f=nlat).values
    if 'prate' in field and (np.array(levs)>=0).all(): 
        cmap = nlcmap(cmap, levs)
        ext = 'neither'
        
    # plot the hovmoller, with time increasing downwards
    x = fcst_xry['lon'].values
    y = fcst_xry.vdates()[::-1]
    hov4plot = np.roll(hov[::-1, :], roll, axis=1)
    csf = ax.contourf(x, y, hov4plot, cmap=cmap, levels=levs, extend=ext)
    if cbar: 
        cb = plt.colorbar(csf, cax=cax)
        cb.set_ticks(levs)
        cb.set_ticklabels(levs)
        
    # make the plot look nice
    xticks = np.arange(0, 361, 60)
    if roll != 0:
        xticklocs = xticks + roll*fcst_xry.dx()
        xticklocs[xticklocs < 0] += 360
        xticklocs[xticklocs > 360] -= 360
    else:
        xticklocs = xticks
    ax.set_xticks(xticklocs)
    xlabs = np.array(['{:3d}W'.format(360-l) if l>=180 else '{:3d}E'.format(l) for l in xticks])
    xx = [i for i,xl in enumerate(xlabs) if xl.strip() in ['180E', '180W']]
    xlabs[xx] = '180'
    xx = [i for i,xl in enumerate(xlabs) if xl.strip() in ['0E', '0W']]
    xlabs[xx] = '0'
    ax.set_xticklabels(xlabs)
    datesFmt = mdates.DateFormatter('%d %b')
    days = mdates.DayLocator()
    ax.yaxis.set_major_locator(days)
    ax.yaxis.set_major_formatter(datesFmt)
    if not show_ylabels: 
        ax.set_yticklabels([])
    ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    if xlims is None:
        ax.set_xlim(x[0], x[-1])
    else:
        ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(y[0], y[-1])
    
    # Set titles
    if units is not None and title is None:
        title = '{} [{}] averaged from {}$^\circ$ to {}$^\circ$'.format(field, units, int(slat), int(nlat))
    if title is not None:
        ax.text(0.0, 1.015, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=14)
    if idate is not None:
        ax.text(1.0, 1.015, 'init: {:%Y-%m-%d %H:00}'.format(idate), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12)
    return csf, hov4plot