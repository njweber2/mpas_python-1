# -*- coding: utf-8 -*-
"""
This package contains functions for plotting some of my most-used
map projections. Each requires an axis (ax) object and returns 
a Basemap (m) object.
"""
from mpl_toolkits.basemap import Basemap
import numpy as np

def draw_projection(ax, proj, mapcol='k'):
    """
    Chooses a projection from the functions below and draw the map.
    """
    if proj=='orthoNP':
    	return draw_ortho_np(ax, mapcol=mapcol)
    elif proj=='mercNP':
    	return draw_merc_np(ax, mapcol=mapcol)
    elif proj=='mercBorneo':
        return draw_merc_borneo(ax, mapcol=mapcol)
    elif proj=='mercMC':
        return draw_merc_mc(ax, mapcol=mapcol)
    elif proj=='mercWarmPool':
        return draw_merc_warmpool(ax, mapcol=mapcol)
    elif proj=='cylCONUS':
        return draw_cyl_conus(ax, mapcol=mapcol)
    elif proj=='cyl_seUS':
        return draw_cyl_seus(ax, mapcol=mapcol)
    elif proj=='stereCONUS':
        return draw_stere_conus(ax, mapcol=mapcol)
    elif proj=='stereNP':
        return draw_stere_np(ax, mapcol=mapcol)
    elif proj=='stereWA':
        return draw_stere_wa(ax, mapcol=mapcol)
    elif proj=='robinGlobal':
        return draw_robin_global(ax, mapcol=mapcol)
    elif proj=='orthoAmerEQ':
        return draw_ortho_americas(ax, mapcol=mapcol)
    elif proj=='mercPNA':
        return draw_merc_pna(ax, mapcol=mapcol)
    else:
    	raise ValueError("Unknown projection {}".format(proj))
    
def draw_ortho_np(ax, mapcol='k'):
    """
    A "from space" view of the North Pacific
    """
    m = Basemap(ax=ax, resolution='c', projection='ortho', lat_0=30.,
                lon_0=180.)
    m.drawcoastlines(linewidth=1.5, color=mapcol)
    m.drawparallels(np.arange(-80,81,20))
    m.drawmeridians(np.arange(0, 360, 20))
    return m
    
def draw_merc_np(ax, mapcol='k'):
    """
    A Mercator projection over the North Pacific
    """
    m = Basemap(ax=ax,projection='merc', llcrnrlat=-0.,urcrnrlat=65.,\
                llcrnrlon=130.,urcrnrlon=250.,\
                rsphere=6371200.,resolution='l',area_thresh=10000)
    m.drawcoastlines(linewidth=1.7, color=mapcol)
    m.drawstates(color=mapcol)
    m.drawcountries(color=mapcol)
    m.drawparallels(np.arange(0.,90,10.),labels=[1,0,0,0],fontsize=10)
    m.drawmeridians(np.arange(0.,360.,20.),labels=[0,0,0,1],fontsize=10)
    return m

def draw_merc_borneo(ax, mapcol='k'):
    """
    A Mercator projection over the Borneo region
    """
    m = Basemap(ax=ax,projection='merc',lon_0=115.,lat_0=90.,lat_ts=0.,\
                llcrnrlat=-10.,urcrnrlat=10.,\
                llcrnrlon=100.,urcrnrlon=130.,\
                rsphere=6371200.,resolution='i',area_thresh=10000)
    m.drawcoastlines(linewidth=1.7, color=mapcol)
    m.drawstates(color=mapcol)
    m.drawcountries(color=mapcol)
    m.drawparallels(np.arange(-90.,90,5.),labels=[1,0,0,0],fontsize=10)
    m.drawmeridians(np.arange(0.,360.,5.),labels=[0,0,0,1],fontsize=10)
    return m

def draw_merc_mc(ax, mapcol='k'):
    """
    A Mercator projection over the Maritime Continent region
    """
    m = Basemap(ax=ax,projection='merc',lon_0=115.,lat_0=90.,lat_ts=0.,\
                llcrnrlat=-25.,urcrnrlat=25.,\
                llcrnrlon=90.,urcrnrlon=170.,\
                rsphere=6371200.,resolution='l',area_thresh=10000)
    m.drawcoastlines(linewidth=1.7, color=mapcol)
    m.drawstates(color=mapcol)
    m.drawcountries(color=mapcol)
    m.drawparallels(np.arange(-90.,90,10.),labels=[1,0,0,0],fontsize=10)
    m.drawmeridians(np.arange(0.,360.,10.),labels=[0,0,0,1],fontsize=10)
    return m

def draw_merc_warmpool(ax, mapcol='k'):
    """
    A Mercator projection over the Indo-Pacific Warm Pool
    """
    m = Basemap(ax=ax,projection='merc',lon_0=125.,lat_0=90.,lat_ts=0.,\
                llcrnrlat=-30.,urcrnrlat=30.,\
                llcrnrlon=50.,urcrnrlon=200.,\
                rsphere=6371200.,resolution='l',area_thresh=10000)
    m.drawcoastlines(linewidth=1.7, color=mapcol)
    m.drawcountries(color=mapcol)
    m.drawparallels(np.arange(-90.,90,10.),labels=[1,0,0,0],fontsize=10)
    m.drawmeridians(np.arange(0.,360.,30.),labels=[0,0,0,1],fontsize=10)
    return m

def draw_cyl_conus(ax, mapcol='k'):
    """
    A cylindrical projection over the continental United States
    """
    m = Basemap(ax=ax,projection='cyl', llcrnrlat=23.,urcrnrlat=53.,\
                llcrnrlon=-127.,urcrnrlon=-66,resolution='l')
    m.drawcoastlines(linewidth=1.7, color=mapcol)
    m.drawstates(color=mapcol)
    m.drawcountries(color=mapcol)
    m.drawparallels(np.arange(-90.,90,10.),labels=[1,0,0,0],fontsize=10)
    m.drawmeridians(np.arange(0.,360.,10.),labels=[0,0,0,1],fontsize=10)
    return m

def draw_cyl_seus(ax, mapcol='k'):
    """
    A cylindrical projection over the southeastern United States
    """
    m = Basemap(ax=ax,projection='cyl', llcrnrlat=25.,urcrnrlat=34.,\
                llcrnrlon=-95.,urcrnrlon=-77,resolution='i')
    m.drawcoastlines(linewidth=1.7, color=mapcol)
    m.drawstates(color=mapcol)
    m.drawcountries(color=mapcol)
    m.drawparallels(np.arange(-90.,90,5.),labels=[1,0,0,0],fontsize=10)
    m.drawmeridians(np.arange(0.,360.,5.),labels=[0,0,0,1],fontsize=10)
    return m

def draw_stere_conus(ax, mapcol='k'):
    """
    A stereographic projection over the continental United States
    """
    m = Basemap(ax=ax,projection='stere', width=5500000,height=3500000,\
                lat_ts=39,lat_0=39,lon_0=-97.,resolution='l')
    m.drawcoastlines(linewidth=1.7, color=mapcol)
    m.drawstates(color=mapcol)
    m.drawcountries(color=mapcol)
    m.drawparallels(np.arange(-90.,90,10.),labels=[1,0,0,0])
    m.drawmeridians(np.arange(0.,360.,10.),labels=[0,0,0,1])
    return m

def draw_stere_wa(ax, mapcol='k'):
    """
    A stereographic projection centered over the Pacific Northwest
    """
    m = Basemap(ax=ax,projection='stere', width=1000000,height=850000,\
                lat_ts=47,lat_0=47,lon_0=-120.,resolution='l')
    m.drawcoastlines(linewidth=1.7, color=mapcol)
    m.drawstates(color=mapcol)
    m.drawcountries(color=mapcol)
    m.drawparallels(np.arange(-90.,90,10.),labels=[1,0,0,0])
    m.drawmeridians(np.arange(0.,360.,10.),labels=[0,0,0,1])
    return m

def draw_stere_np(ax, mapcol='k'):
    """
    A stereographic projection over the North Pacific
    """
    m = Basemap(ax=ax,projection='stere', width=8000000,height=5700000,\
                lat_ts=40,lat_0=40,lon_0=-160,resolution='l')
    m.drawcoastlines(linewidth=1.7, color=mapcol)
    m.drawstates(color=mapcol)
    m.drawcountries(color=mapcol)
    m.drawparallels(np.arange(-90.,90,10.),labels=[0,0,0,0])
    m.drawmeridians(np.arange(0.,360.,10.),labels=[0,0,0,0])
    return m

def draw_robin_global(ax, mapcol='k'):
    """
    A stereographic projection over the North Pacific
    """
    m = Basemap(ax=ax,projection='robin', lon_0=180, resolution='l')
    m.drawcoastlines(linewidth=1.2, color=mapcol)
    m.drawparallels(np.arange(-90.,90,30.),labels=[1,0,0,0])
    m.drawmeridians(np.arange(0.,360.,60.),labels=[0,0,0,1])
    return m

def draw_ortho_americas(ax, mapcol='k'):
    """
    An equator-centered  orthographic projection at 90W 
    """
    m = Basemap(ax=ax,projection='ortho', lat_0=0, lon_0=-90, resolution='l')
    m.drawcoastlines(linewidth=1.2, color=mapcol)
    m.drawparallels(np.arange(-90.,90,30.),labels=[0,0,0,0])
    m.drawmeridians(np.arange(0.,360.,30.),labels=[0,0,0,0])
    return m

def draw_merc_pna(ax, mapcol='k'):
    """
    A Mercator projection over the PNA teleconnection region (WPac --> Eastern US)
    """
    m = Basemap(ax=ax,projection='merc',lon_0=200.,lat_0=90.,lat_ts=0.,\
                llcrnrlat=0.,urcrnrlat=75.,\
                llcrnrlon=100.,urcrnrlon=310.,\
                rsphere=6371200.,resolution='l',area_thresh=10000)
    m.drawcoastlines(linewidth=1.7, color=mapcol)
    m.drawparallels(np.arange(-90.,90,30.),labels=[1,0,0,0],fontsize=10)
    m.drawmeridians(np.arange(0.,360.,30.),labels=[0,0,0,1],fontsize=10)
    return m
