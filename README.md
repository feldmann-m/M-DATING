Mesocyclone detection for MeteoSwiss operational radar data

Algorithm description: https://wcd.copernicus.org/articles/2/1225/2021/; https://www.nature.com/articles/s41612-023-00352-z

Algorithm citation:
Feldmann, M., Germann, U., Gabella, M., and Berne, A.: A characterisation of Alpine mesocyclone occurrence, Weather Clim. Dynam., 2, 1225–1244, https://doi.org/10.5194/wcd-2-1225-2021, 2021. 
Feldmann, M., Hering, A., Gabella, M. and Berne, A.: Hailstorms and rainstorms versus supercells—a regional analysis of convective storm types in the Alpine region, npj Clim Atmos Sci, 6, 19, https://doi.org/10.1038/s41612-023-00352-z, 2023.

This algorithm detects cyclonic and anticyclonic supercells based on provided thunderstorm contours in a Cartesian framework, and Doppler velocity data in polar coordinates. The resulting objects are returned as a point-based object in a json file.
The script realtime_parallel executes a single timestep of mesocyclone detection, using parallel computation. realtime_serial uses serial computation and takes significantly longer. To obtain timeseries of results, the realtime script needs to be called for every desired timestep.
The script realtime_plot provides a visualization of a single timestep. In addition, daily_plot provides a daily summary (based on the date format of the files).
script.sh provides an example, how to call the functions.
The folder "library" contains all internal functions of the algorithm. "mask_data" contains the lookup tables for the Swiss radar network to speed up coordinate transformations.

