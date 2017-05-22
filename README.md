mpas_python
===========

A few Python scripts for plotting output from the MPAS weather model
All scripts are in the scripts subdirectory.

--> mpasoutput.py --> Contains the MPASprocessed and MPASraw classes, which are xarray-based datasets for storing MPAS forecast output. MPASprocessed is for MPAS forecasts on a lat/lon grid (e.g., converted to lat/lon coordinates by Michael Duda's convert_mpas utility: https://github.com/mgduda/convert_mpas/). MPASraw is for storing MPAS output on its native Voronoi cell/edge/vertex grid.

--> plotting_mpas_latlon.py --> A module containing tools for plotting MPAS forecast output stored in the MPASprocessed class (regular lat/lon grid).

--> plotting_mpas_mesh.py --> A module containing tools for plotting MPAS forecast output stored in the MPASraw (native mesh) class.

--> verification.py --> Contains tools for loading other geophysical datasets and using them to verify MPAS forecasts. Currently only handles 3-hourly GFS analyses.

--> map_projections.py --> A rough module containing a number of pre-configured Basemap projections for quick access.

--> old/ --> Contains the original scripts written by Luke Madaus (modified by N. Weber to be compatible with python 3.x and the xarray-based classes) that plot raw MPAS output on the Voronoi mesh and a lat/lon grid.

--> gfs_verification.table --> needed for converting GFS gribs to netcdf with the wgrib2 utility

The jupyter notebooks contain a few examples of how to use these tools. Note that the colormaps are retrieved from a local package that is not included here.


To Do:
==========
--> Add functionality for a 4D grid (new dimension: z)

--> Add more plotting functions to a) plotting_mpas_mesh.py and b) plotting_mpas_latlon.py

--> Rename classes in mpasoutput

--> Add operational and reforecast CFSv2 capabilities

--> Add NCEP reanalyses to the verification datasets

--> Start writing MJO/tropical convection evaluation tools

--> Write method for temporally averaging the forecasts (e.g., week-1, week-2,...)

