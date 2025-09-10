"""
======================================
Reproject IRIS SJI (rolled) to SDO/AIA
======================================

In this example we will show how to reproject a rolled IRIS dataset to SDO/AIA.

The IRIS team at LMSAL provides AIA data cubes which are coaligned to the IRIS FOV
for each observation from https://iris.lmsal.com/search/

Therefore this example is more a showcase of functionally.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from astropy.wcs.utils import wcs_to_celestial_frame

import sunpy.map
from aiapy.calibrate import update_pointing
from aiapy.calibrate.util import get_pointing_table
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.visualization.drawing import extent

from irispy.io import read_files
from irispy.obsid import ObsID

###############################################################################
# We start with getting the data.
# This is done by downloading the data from the IRIS archive.
#
# In this case, we will use ``pooch`` as to keep this example self-contained
# but using your browser will also work.

sji_filename = pooch.retrieve(
    "http://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2014/09/19/20140919_051712_3860608353/iris_l2_20140919_051712_3860608353_SJI_2832_t000.fits.gz",
    known_hash="02a1cdbe2014e24b04ff782dcfdbaf553c5a03404813888ddea8c50a9d6b2630",
)

###############################################################################
# We will now open the slit-jaw imager (SJI) file we just downloaded.

sji_2832 = read_files(sji_filename)
# Printing will give us an overview of the file.
print(sji_2832)
# ``.meta`` contains the entire FITS header from the primary HDU.
# Since it is very long, we won't actually print it here.

###############################################################################
# Can't remember what is OBSID 3860608353?
# **irispy** has an utility function that will provide more information.

print(ObsID(sji_2832.meta["OBSID"]))

###############################################################################
# We also have the option of going directly to an individual scan.

sji_cut = sji_2832[45]
print(sji_cut)

###############################################################################
# We need to get the coordinate frame for the IRIS data.
# While this is stored in the WCS, getting a coordinate frame is a little more involved.

sji_frame = wcs_to_celestial_frame(sji_cut.basic_wcs)

###############################################################################
# This dataset has a peculiarity: the observation has a 45 degree roll.
# The image does not have a 45 degree rotation because plotting shows the data
# in the way they are written in the file.
# We will a coordinate grid to make this clear.
# You can also change the axis labels and ticks if you so desire.
# `WCSAxes provides us an API we can use. <https://docs.astropy.org/en/stable/visualization/wcsaxes/index.html>`__

plt.figure()
ax = sji_cut.plot()
plt.title(f"IRIS SJI {sji_2832.meta['TWAVE1']}", pad=20)
ax.coords.grid()

###############################################################################
# We will want to align the data to AIA.
# First we will want to pick a timestamp during the observation.
#
# Lets us now find the SJI observation where the time is closest to 06:00 on 2014-09-19.

(time_sji,) = sji_2832.axis_world_coords("time")
time_target = Time("2014-09-19T06:00:00.0")
time_index = np.abs(time_sji - time_target).argmin()
time_stamp = time_sji[time_index].isot
print(time_index, time_stamp)

###############################################################################
# The fact that it is rolled 45 degrees makes manual alignment tricky
# and will illustrate the usefulness of working with WCS.
# We will download an AIA 170 nm image from the VSO.
# Once we have acquired it, we will need to use **aiapy** to prep this image.

search_results = Fido.search(
    a.Time(time_stamp, Time(time_stamp) + TimeDelta(1 * u.minute), near=time_stamp),
    a.Instrument.aia,
    a.Wavelength(1700 * u.AA),
)
files = Fido.fetch(search_results, site="NSO")
aia_map = sunpy.map.Map(files[0])
pointing_table = get_pointing_table(
    source="JSOC",
    time_range=(Time(time_stamp) - TimeDelta(5 * 60 * u.minute), Time(time_stamp) + TimeDelta(1 * u.minute)),
)
aia_map = update_pointing(aia_map, pointing_table=pointing_table)
# You don't need to register AIA images unless you need them aligned to other AIA images.
# otherwise you are degrading the data as the affine transform is not perfect.
# But it is the last step to get a level 1.5 image.

###############################################################################
# Now let's plot the IRIS field of view on the AIA image.
#
# This IRIS data has no observer coordinate information
# **irispy** will set this to be at Earth.
# This will allow us to transform from IRIS to any another observer.
#
# Using :meth:`sunpy.map.GenericMap.draw_extent`, drawing regions is straightforward.

aia_bottom_left = SkyCoord(-850 * u.arcsec, -50 * u.arcsec, frame=aia_map.coordinate_frame)
aia_top_right = SkyCoord(-650 * u.arcsec, 150 * u.arcsec, frame=aia_map.coordinate_frame)
aia_sub = aia_map.submap(aia_bottom_left, top_right=aia_top_right)

fig = plt.figure()
ax = plt.subplot(projection=aia_sub)
aia_sub.plot()
extent(ax, sji_cut.basic_wcs)

###############################################################################
# We have a green square showing the region of the IRIS observation.
# To work with both IRIS and AIA data, it helps if the image axes are aligned,
# and for this we need to rotate one of them. We can either rotate SDO/AIA to the
# IRIS frame, or vice-versa.
#
# We will rotate the AIA data, using `sunpy.map.GenericMap.reproject_to`.
# As `sunpy` does not support gWCS (yet), we have to use the basic WCS.

aia_reprojected = aia_sub.reproject_to(sji_cut.basic_wcs)

###############################################################################
# Now we can see the results.

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection=aia_reprojected.wcs)
aia_reprojected.plot(axes=ax1)
ax1.set_title("")

ax2 = fig.add_subplot(1, 2, 2, projection=sji_cut.basic_wcs)
sji_cut.plot(axes=ax2)

fig.tight_layout()

###############################################################################
# Finally, one way to visualize the alignment is to plot the AIA contours on the IRIS SJI image.

fig = plt.figure()

ax1 = fig.add_subplot(111, projection=sji_cut.wcs)
sji_cut.plot(axes=ax1)
aia_reprojected.draw_contours(levels=[500], colors=["red"], linewidths=2)
ax1.set_title("IRIS SJI with AIA contours")

plt.show()
