"""
========================================
Potential Offset between SJI and SG data
========================================

In this example, we are showcase a potential offset in WCS coordinates between IRIS SJI and IRIS SG data.
The offset varies between observations and is not fixed.

The cause is unknown at this time and this has not been cross-checked with SSWIDL.
So it is possible this is just a bug in the Python library.

Comment from author:
Running this locally gives a different outcome than from the final version online and I have been unable to figure out why.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import time_support
from astropy.wcs.utils import wcs_to_celestial_frame

from irispy.io import read_files

time_support()

###############################################################################
# We will start by getting some data from the IRIS archive.
#
# In this case, we will use ``pooch`` so to keep this example self-contained
# but using your browser will also work.
#
# Using the url: https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20130902_182935_4000005156_2013-09-02T18%3A29%3A352013-09-02T18%3A29%3A35.xml
# we are after the 2796 Slit-Jaw and the raster observation.

raster_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2013/09/02/20130902_182935_4000005156/iris_l2_20130902_182935_4000005156_raster.tar.gz",
    known_hash="91211a52e278fb6e535242d4d6064facf9f93cf24f0a433c276ace1b2d621e7d",
)
sji_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2013/09/02/20130902_182935_4000005156/iris_l2_20130902_182935_4000005156_SJI_2796_t000_deconvolved.fits.gz",
    known_hash="2043c85e567a5a0ad8cf9024745dae87f632d5c4cd019cc0e7bfeb7a1fae99d0",
)

###############################################################################
# Now to open the files using ``irispy``.

raster = read_files(raster_filename)
sji_2796 = read_files(sji_filename)

###############################################################################
# Now we will find the closest SJI time to the 56th raster step.

c_ii = raster["C II 1336"]

times_SG = c_ii.axis_world_coords("time", wcs=c_ii.extra_coords)
(time_2796,) = sji_2796.axis_world_coords("time")
# We picked randomly.
raster_idx = 55
time_target = times_SG[0][raster_idx]

time_idx_2796 = np.abs(time_2796 - time_target).argmin()
time_stamp_2796 = time_2796[time_idx_2796].isot
print(time_stamp_2796, "\n", time_target, "\n", time_idx_2796)

###############################################################################
# We will require the auxiliary data from both files later on.

# The raster file is extracted to the cache directory via pooch, so we need to do some magic to get it:
raster_aux_data = fits.getdata(
    next(iter(pooch.os_cache("pooch").expanduser().glob("*iris_l2_20130902_182935_4000005156_raster/*fits"))), ext=-2
)
sji_aux_data = fits.getdata(sji_filename, ext=-2)

# If you check the headers, you will find an index for POFFXSJI and POFFYSJI:
# SJI X (spectral direction) shift in pixels
# SJI Y (spatial direction) shift in pixels
sji_offset = sji_aux_data[time_idx_2796, 29] * 0.167 * u.arcsec
sg_offset = raster_aux_data[raster_idx, 45] * 0.167 * u.arcsec

# Also in the SJI AUX data is the pixel location of the slit
# We will need to get the world location later on.
sji_slit_location_pixel_x = sji_aux_data[time_idx_2796, 4]
sji_slit_location_pixel_y = sji_aux_data[time_idx_2796, 5]

###############################################################################
# We can now get the slit location from the SG cube and apply the offsets to the location.

sji_2796_closest = sji_2796[time_idx_2796]
sji_2796_frame = wcs_to_celestial_frame(sji_2796_closest.basic_wcs)

raster_lon_coords = c_ii.axis_world_coords_values("custom:pos.helioprojective.lon")[0][raster_idx]
raster_lat_coords = c_ii.axis_world_coords_values("custom:pos.helioprojective.lat")[0][raster_idx]

# Now we will get the raster location from its WCS and overlay that on the SJI below.
slit = SkyCoord(Tx=raster_lon_coords, Ty=raster_lat_coords, frame=sji_2796_frame)
# In this case, the offsets are negative, so we add them instead.
slit_with_sg_offset = SkyCoord(Tx=raster_lon_coords + (-1 * sg_offset), Ty=raster_lat_coords, frame=sji_2796_frame)
slit_with_sji_offset = SkyCoord(Tx=raster_lon_coords + (-1 * sji_offset), Ty=raster_lat_coords, frame=sji_2796_frame)

###############################################################################
# Now we can visualize the difference.

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(projection=sji_2796_closest)
sji_2796_closest.plot(ax)

# This is the pixel location of the slit in the SJI data based on the axillary data
slit_location_from_sji_aux = sji_2796[time_idx_2796].wcs.pixel_to_world(
    sji_slit_location_pixel_x, sji_slit_location_pixel_y
)
ax.plot_coord(slit_location_from_sji_aux, ".", color="white", label="Slit Pixel Location")

# Now these are the slit locations (with the last two modified by offset values in the axillary data)
ax.plot_coord(slit, color="white", linestyle="--", linewidth=1, label="Slit WCS")
ax.plot_coord(slit_with_sg_offset, color="red", linestyle="-", linewidth=1, label="Slit WCS + SG Offset")
ax.plot_coord(slit_with_sji_offset, color="green", linestyle="-", linewidth=1, label="Slit WCS + SJI Offset")
ax.legend()

# "Crop" the image without touching the actual data.
bbox = SkyCoord([140, 180] * u.arcsec, [50, 90] * u.arcsec, frame=sji_2796_frame)
x_limit, y_limit = sji_2796_closest.wcs.world_to_pixel(bbox)
ax.set_xlim(int(x_limit[0]), int(x_limit[1]))
ax.set_ylim(int(y_limit[0]), int(y_limit[1]))

plt.show()
