"""
=============================
Co-align IRIS SJI  to SDO/AIA
=============================

In this example we will show how to co-align a rolled IRIS dataset to SDO/AIA.

The IRIS team at LMSAL provides AIA data cubes which are coaligned to the IRIS FOV
for each observation from https://iris.lmsal.com/search/

Therefore this example is more a showcase of functionally.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch
from sunkit_image.coalignment import coalign

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

import sunpy.map
from aiapy.calibrate import update_pointing
from aiapy.calibrate.util import get_pointing_table
from sunpy.net import Fido
from sunpy.net import attrs as a

from irispy.io import read_files

###############################################################################
# We start with getting the data.
# This is done by downloading the data from the IRIS archive.
#
# In this case, we will use ``pooch`` as to keep this example self-contained
# but using your browser will also work.

sji_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2025/07/10/20250710_121126_3893010094/iris_l2_20250710_121126_3893010094_SJI_2832_t000_deconvolved.fits.gz",
    known_hash="a2cc80da9fedc27add3f04abd126894d546059ea468d3ab5adbcc482e40e506a",
)

###############################################################################
# We will now open the slit-jaw imager (SJI) file we just downloaded.

sji_2832 = read_files(sji_filename)

###############################################################################
# We will want to align the data to AIA.
# First we will want to pick a timestamp during the observation.

(time_sji,) = sji_2832.axis_world_coords("time")
# We get a sunpy map as the coalignment works on sunpy maps only for now.
sji_map = sji_2832.to_maps(8)

###############################################################################
# We will download the closest AIA 170 nm image from the VSO.
# Once we have acquired it, we will need to use **aiapy** to prep this image.

search_results = Fido.search(
    a.Time(time_sji[0], Time(time_sji[0]) + TimeDelta(1 * u.minute), near=time_sji[0]),
    a.Instrument.aia,
    a.Wavelength(1700 * u.AA),
)
files = Fido.fetch(search_results, site="NSO")
aia_map = sunpy.map.Map(files[0])
pointing_table = get_pointing_table(
    source="JSOC",
    time_range=(Time(time_sji[0]) - TimeDelta(5 * 60 * u.minute), Time(time_sji[0]) + TimeDelta(1 * u.minute)),
)
aia_map = update_pointing(aia_map, pointing_table=pointing_table)

# Crop the AIA FOV to be similar to IRIS.
aia_crop = aia_map.submap(
    bottom_left=SkyCoord(
        sji_map.bottom_left_coord.Tx - 50 * u.arcsec,
        sji_map.bottom_left_coord.Ty - 50 * u.arcsec,
        frame="helioprojective",
        observer=sji_map.bottom_left_coord.observer,
    ),
    top_right=SkyCoord(
        sji_map.top_right_coord.Tx + 50 * u.arcsec,
        sji_map.top_right_coord.Ty + 50 * u.arcsec,
        frame="helioprojective",
        observer=sji_map.top_right_coord.observer,
    ),
)
# Before co-aligning the images, we first resample the AIA image to the same plate
# scale as the IRIS image. This will ensure better results from our coalignment.
nx = (aia_crop.scale.axis1 * aia_crop.dimensions.x) / sji_map.scale.axis1.to(u.arcsec / u.pix)
ny = (aia_crop.scale.axis2 * aia_crop.dimensions.y) / sji_map.scale.axis2.to(u.arcsec / u.pix)
aia_upsampled = aia_crop.resample(u.Quantity([nx, ny]))

# ###############################################################################
# # One way to visualize the alignment is to plot the AIA contours on the IRIS SJI image.

fig = plt.figure()

ax1 = fig.add_subplot(111, projection=sji_map.wcs)
sji_map.plot(axes=ax1, autoalign=True)
aia_upsampled.draw_contours(levels=[500] * u.DN, colors=["red"], linewidths=2)
ax1.set_title("IRIS SJI with AIA contours")

###############################################################################
# As one can see, the alignment is not perfect. Creating a pixel perfect WCS
# is very difficult due to uncertainties in locations and the pointing information.
#
# So what we can do is a cross-correlation between IRIS and SDO/AIA to see if we can
# improve this. The following uses ``sunkit-image`` and currently only works on sunpy Maps,
# so we will use the SJI Map for this case and not the cube.
#
# Before co-aligning the images, we have to make sure that both images have the
# image scale, as this is important for the routine.
#
# Now we can co-align cross-correlation using the "match_template" method.
# For details of the implementation refer to the documentation of
# `~sunkit_image.coalignment.match_template.match_template_coalign`.

# We need to prepare the SJI data.
sji_map_corrected_data = sji_map.data.copy()
nan_mask = ~np.isfinite(sji_map_corrected_data)
if np.any(nan_mask):
    sji_map_corrected_data[nan_mask] = 0
sji_map_corrected = sunpy.map.Map(sji_map_corrected_data, sji_map.meta)

coaligned_sji_map = coalign(sji_map_corrected, aia_upsampled, method="match_template", pad_input=True)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection=coaligned_sji_map.wcs)
coaligned_sji_map.plot(axes=ax1, autoalign=True)
aia_upsampled.draw_contours(levels=[500] * u.DN, colors=["red"], linewidths=2)
ax1.set_title("Co-aligned IRIS SJI with AIA contours")

plt.show()
