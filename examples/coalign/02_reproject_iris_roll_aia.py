"""
======================================
Reproject IRIS SJI (rolled) to SDO/AIA
======================================

In this example we will show how to reproject a rolled IRIS dataset to SDO/AIA.

The IRIS team at LMSAL provides AIA data cubes which are co-aligned to the IRIS FOV for
each observation via the `IRIS data search page <https://iris.lmsal.com/search/>`__.

Therefore this example is more of a showcase of functionality.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

import sunpy.map
from aiapy.calibrate import update_pointing
from aiapy.calibrate.utils import get_pointing_table
from sunpy.visualization.drawing import extent

from irispy.io import read_files
from irispy.obsid import ObsID

###############################################################################
# `We start with getting data from the IRIS data archive <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20140919_051712_3860608353_2014-09-19T05%3A17%3A122014-09-19T05%3A17%3A12.xml>`__.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.
#
# You will need to update the path to the data in the next section if you do that.

sji_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2014/09/19/20140919_051712_3860608353/iris_l2_20140919_051712_3860608353_SJI_2832_t000.fits.gz",
    known_hash="7ec0f3d63d97bc7620675c78fb6c670ef5b4249d31ef7818435b629c04b72f60",
)

###############################################################################
# We will now open the data using a helper function which is designed to read
# all files from a single observation.

sji_2832 = read_files(sji_filename)

###############################################################################
# Printing will give us an overview of the file.

print(sji_2832)
# ``.meta`` holds the full primary FITS header, which is too long to print here.

###############################################################################
# Can't remember what OBSID 3860608353 is? ``irispy.obsid.ObsID`` describes it.

print(ObsID(sji_2832.meta["OBSID"]))

###############################################################################
# We will want to align the data to AIA, so we pick one frame of the observation:
# the one closest to 06:00 on 2014-09-19.

(time_sji,) = sji_2832.axis_world_coords("time")
time_target = Time("2014-09-19T06:00:00.0")
time_index = np.abs(time_sji - time_target).argmin()
time_stamp = time_sji[time_index].isot
print(time_index, time_stamp)

###############################################################################
# We can go directly to that individual frame.

sji_cut = sji_2832[time_index]
print(sji_cut)

###############################################################################
# We will need the coordinate frame of the IRIS data, which the cube provides as
# ``celestial_frame``.

sji_frame = sji_cut.celestial_frame

###############################################################################
# This observation has a 45 degree roll. The image is not rotated because plotting
# shows the data as they are stored in the file, so we add a coordinate grid to make
# the roll clear.

plt.figure()
ax = sji_cut.plot()
plt.title(f"IRIS SJI {sji_2832.meta['TWAVE1']:.0f} Å", pad=20)
# WCSAxes needs ``grid_type="contours"`` to draw the grid of this WCS correctly.
ax.coords.grid(grid_type="contours")

###############################################################################
# The 45 degree roll makes manual alignment tricky and shows the usefulness of working
# with WCS. We need the AIA 170 nm image closest to that time, which you can find and
# download from the VSO with `sunpy.net.Fido`:
#
# .. code-block:: python
#
#     from sunpy.net import Fido
#     from sunpy.net import attrs as a
#
#     search_results = Fido.search(
#         a.Time(time_stamp, Time(time_stamp) + TimeDelta(1 * u.minute), near=time_stamp),
#         a.Instrument.aia,
#         a.Wavelength(1700 * u.AA),
#     )
#     files = Fido.fetch(search_results, site="NSO")
#
# To keep this example independent of the VSO, we download the file this search
# returns from `irispy-data <https://github.com/LM-SAL/irispy-data>`__ instead.
# Once we have it, we will need to use **aiapy** to prep this image.

aia_filename = pooch.retrieve(
    "https://github.com/LM-SAL/irispy-data/releases/download/v1/aia.lev1.1700A_2014_09_19T05_59_18.71Z.image_lev1.fits",
    known_hash="f36224ec4da14259aa90abdb6e53e4d5986b357ac45e96574c068a9dc3a5dceb",
)
aia_map = sunpy.map.Map(aia_filename)
pointing_table = get_pointing_table(
    source="JSOC",
    time_range=(Time(time_stamp) - TimeDelta(5 * 60 * u.minute), Time(time_stamp) + TimeDelta(1 * u.minute)),
)
aia_map = update_pointing(aia_map, pointing_table=pointing_table)
# We skip registering the image, the last step to level 1.5, which is only needed to
# align AIA images with each other and resamples the data.

###############################################################################
# Now let us draw the IRIS field of view on the AIA image with
# `sunpy.visualization.drawing.extent`. This IRIS file has no observer information, so
# ``irispy`` places the observer at Earth, which lets us transform to other observers.

aia_bottom_left = SkyCoord(-850 * u.arcsec, -50 * u.arcsec, frame=aia_map.coordinate_frame)
aia_top_right = SkyCoord(-650 * u.arcsec, 150 * u.arcsec, frame=aia_map.coordinate_frame)
aia_sub = aia_map.submap(aia_bottom_left, top_right=aia_top_right)

fig = plt.figure()
ax = plt.subplot(projection=aia_sub)
aia_sub.plot()
extent(ax, sji_cut.fits_wcs)

###############################################################################
# The outline shows the IRIS field of view, rotated by its 45 degree roll. To work with
# both datasets, it helps to align the image axes, so we rotate the AIA data onto the
# IRIS grid with `sunpy.map.GenericMap.reproject_to`. As `sunpy` does not support gWCS
# (yet), we use ``fits_wcs``.

aia_reprojected = aia_sub.reproject_to(sji_cut.fits_wcs)

###############################################################################
# Finally, one way to visualize the alignment is to plot the AIA contours on the IRIS SJI image.

fig = plt.figure()

ax1 = fig.add_subplot(111, projection=sji_cut.wcs)
sji_cut.plot(axes=ax1)
aia_reprojected.draw_contours(levels=[500], colors=["red"], linewidths=2)
ax1.set_title("IRIS SJI with AIA contours")

plt.show()

###############################################################################
# The reprojection alone does not align the two images, because the pointing was not
# accurate to begin with. :ref:`sphx_glr_generated_gallery_coalign_01_coalign_iris_aia.py`
# shows how to co-align them.

# sphinx_gallery_thumbnail_number = 3
