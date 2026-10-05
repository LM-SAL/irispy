"""
============================
Manipulate spectrograph data
============================

In this example, we will showcase how to open, crop and plot IRIS spectrograph data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch

import astropy.units as u
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.visualization import quantity_support
from astropy.wcs.utils import wcs_to_celestial_frame

from sunpy.coordinates.frames import Helioprojective

from irispy.io import read_files
from irispy.utils.moments import average_window

quantity_support()

###############################################################################
# `We start with getting data from the IRIS data archive <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20180102_153155_3610108077_2018-01-02T15%3A31%3A552018-01-02T15%3A31%3A55.xml>`__.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.
#
# You will need to update the path to the data in the next section if you do that.

raster_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2018/01/02/20180102_153155_3610108077/iris_l2_20180102_153155_3610108077_raster.tar.gz",
    known_hash="8949562149cfa5fba067b5b102e8434b14cea3c3416dd79c06b7f6e211c61a39",
)

###############################################################################
# We will now open the data using a helper function which is designed to read
# all files from a single observation.

raster = read_files(raster_filename)

###############################################################################
# Let us now explore what was returned. Printing gives an overview of the raster collection.

print(raster)

###############################################################################
# The keys are the spectral windows.

print(raster.keys())

###############################################################################
# We can get the Mg II k window:

mg_ii = raster["Mg II k 2796"]
print(mg_ii)

###############################################################################
# This is an `irispy.spectrograph.SpectrogramCubeSequence` which contains each
# complete raster as one individual `irispy.spectrograph.SpectrogramCube` object.
# In this case, there was only one complete raster, so the first axis is only length 1.
#
# So we will index to get the first raster and work with that.

mg_ii = mg_ii[0]
print(mg_ii)

###############################################################################
# Now we have more information about the data, including the OBS ID and description.
#
# Let's plot it:

fig = plt.figure()
# ``get_animation`` lets Sphinx Gallery render the sequence as an animation.
raster_animation = mg_ii.plot(fig=fig).get_animation()

###############################################################################
# If we want to "raster" over wavelength, we can do the following.
#
# This will also "transpose" the data but this is only for visualization purposes.
# The first wavelength pixels of this window hold almost no data (they are fill values,
# which are masked), so we crop to the Mg II k line (279.4 to 279.9 nm) and the animation
# starts on real data. We also set the vmin and vmax, as by default "plot" works them out
# from the first slice only.

mg_ii_k_line = mg_ii.crop([SpectralCoord(279.4, unit=u.nm), None], [SpectralCoord(279.9, unit=u.nm), None])
fig = plt.figure()
wavelength_animation = mg_ii_k_line.plot(fig=fig, plot_axes=["x", "y", None], vmin=0, vmax=500).get_animation()

###############################################################################
# We can plot a spectrum at raster step 120 and slit pixel 200 using the data directly.
# We read the wavelengths of the Mg window by calling
# `ndcube.NDCube.axis_world_coords` for "wl" (wavelength).

(mg_wave,) = mg_ii.axis_world_coords("wl")

fig, ax = plt.subplots()
# The array still holds the fill values (-200) that the cube masks, so we hide them as well.
ax.plot(mg_wave.to("AA"), np.where(mg_ii.mask[120, 200], np.nan, mg_ii.data[120, 200]))
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel(f"Intensity [{mg_ii.unit}]")

###############################################################################
# Using the data directly loses the metadata, the mask and the WCS, so prefer the WCS
# wherever possible and only use the data for custom processing.
#
# If you are unfamiliar with WCS, see the `astropy WCS <https://docs.astropy.org/en/stable/wcs/index.html>`__
# and `WCSAxes <https://docs.astropy.org/en/stable/visualization/wcsaxes/index.html>`__ documentation, and
# `ndcube's guide to coordinates <https://docs.sunpy.org/projects/ndcube/en/stable/explaining_ndcube/coordinates.html>`__.
#
# For example, which wavelength pixel corresponds to the Mg II k core (279.63 nm)?

iris_observer = wcs_to_celestial_frame(mg_ii.wcs.celestial).observer
iris_frame = Helioprojective(observer=iris_observer)
wcs_loc = mg_ii.wcs.world_to_pixel(
    SpectralCoord(279.63, unit=u.nm),
    SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=iris_frame),
)
mg_index = int(np.round(wcs_loc[0]))
print(mg_index)

###############################################################################
# Now we will plot a spectroheliogram at the Mg II k core wavelength, by cropping
# with a `~astropy.coordinates.SpectralCoord`.

# The bounds are in world axis order and ``None`` means that axis is not cropped.
lower_corner = [SpectralCoord(279.63, unit=u.nm), None]
upper_corner = [SpectralCoord(279.63, unit=u.nm), None]
mg_spec_crop = mg_ii.crop(lower_corner, upper_corner)

fig = plt.figure()
ax = fig.add_subplot(111, projection=mg_spec_crop.wcs)
# Put the raster steps along x, so that longitude runs horizontally.
mg_spec_crop.plot(axes=ax, plot_axes=["x", "y"])

###############################################################################
# `~irispy.utils.moments.average_window` averages the samples in a wavelength window
# into a map; here 279.4 to 279.9 nm around the Mg II k core.

mg_ii_k_map = average_window(mg_ii, [279.4, 279.9] * u.nm)

fig = plt.figure()
ax = fig.add_subplot(111, projection=mg_ii_k_map.wcs)
mg_ii_k_map.plot(axes=ax, plot_axes=["x", "y"])

###############################################################################
# Imagine there's a really cool feature at (-338", 275"), how can you plot
# the spectrum at that location?

lower_corner = [None, SkyCoord(-338 * u.arcsec, 275 * u.arcsec, frame=iris_frame)]
upper_corner = [None, SkyCoord(-338 * u.arcsec, 275 * u.arcsec, frame=iris_frame)]
mg_ii_cut = mg_ii.crop(lower_corner, upper_corner)

fig = plt.figure()
ax = fig.add_subplot(111, projection=mg_ii_cut.wcs)
mg_ii_cut.plot(axes=ax)

plt.show()

###############################################################################
# You may also want to know when this observation was taken. ``.meta`` describes the
# observation as a whole.

print(mg_ii.meta)

###############################################################################
# The time of each exposure (raster step) is available as ``.time``.

print(mg_ii.time)

# sphinx_gallery_thumbnail_number = 4
