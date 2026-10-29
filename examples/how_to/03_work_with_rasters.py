"""
============================
Manipulate spectrograph data
============================

In this example, we will showcase how to open, crop and plot IRIS spectrograph data.
"""

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.visualization import quantity_support

from irispy.data import sample
from irispy.io import read_files
from irispy.utils.moments import average_window

quantity_support()

###############################################################################
# We start with the raster sample file, which ``irispy.data.sample`` downloads on first use.
# To use your own data, pass the path of a Level 2 raster tarball from the
# `IRIS data archive <https://www.lmsal.com/hek/hcr>`__ to ``read_files`` instead.

raster_filename = sample.RASTER_TAR

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

###############################################################################
# This observation has five complete rasters. Each spectral window is one
# `irispy.spectrograph.SpectrogramCube` with raster, raster step, slit and
# wavelength axes, so ``mg_ii[0]`` is the first raster. We work with that one.

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

mg_ii_k_line = mg_ii.crop(
    [SpectralCoord(279.4, unit=u.nm), None, None, None], [SpectralCoord(279.9, unit=u.nm), None, None, None]
)
fig = plt.figure()
wavelength_animation = mg_ii_k_line.plot(fig=fig, plot_axes=["x", "y", None], vmin=0, vmax=100).get_animation()

###############################################################################
# We can plot a spectrum at raster step 8 and slit pixel 200 using the data directly.
# We read the wavelengths of the Mg window by calling
# `ndcube.NDCube.axis_world_coords` for "wl" (wavelength).

(mg_wave,) = mg_ii.axis_world_coords("wl")

fig, ax = plt.subplots()
# The array still holds the fill values (-200) that the cube masks, so we hide them as well.
ax.plot(mg_wave.to("AA"), np.where(mg_ii.mask[8, 200], np.nan, mg_ii.data[8, 200]))
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

iris_frame = mg_ii.celestial_frame
wcs_loc = mg_ii.fits_wcs.world_to_pixel(
    SpectralCoord(279.63, unit=u.nm),
    SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=iris_frame),
)
mg_index = int(np.round(wcs_loc[0]))
print(mg_index)

###############################################################################
# Now we will plot a spectroheliogram at the Mg II k core wavelength, by cropping
# with a `~astropy.coordinates.SpectralCoord`.

# A crop takes one entry per world coordinate of the cube: the wavelength, sky position,
# time and raster step, plus the raster scan for the cube of a whole window. ``None``
# leaves that coordinate uncropped. The sky position, time and raster step (and scan)
# must be given together. The same bound twice selects a single wavelength.
k_core = [SpectralCoord(279.63, unit=u.nm), None, None, None]
mg_spec_crop = mg_ii.crop(k_core, k_core)

###############################################################################
# By default, the plot shows only spatial coordinates. Include ``"time"`` in
# ``axes_coordinates`` to add the exposure times to the Mg II k core spectroheliogram.
# We put the raster steps along x, so that longitude and time run horizontally.

fig = plt.figure()
mg_spec_crop.plot(
    plot_axes=["x", "y"],
    axes_coordinates=["custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat", "time"],
)
fig.tight_layout()

###############################################################################
# `~irispy.utils.moments.average_window` averages the samples in a wavelength window
# into a map; here 279.4 to 279.9 nm around the Mg II k core.

mg_ii_k_map = average_window(mg_ii, [279.4, 279.9] * u.nm)

fig = plt.figure()
ax = fig.add_subplot(111, projection=mg_ii_k_map.wcs)
mg_ii_k_map.plot(axes=ax, plot_axes=["x", "y"])

###############################################################################
# Imagine there's a really cool feature at (-320", 400"), how can you plot
# the spectrum at that location?

# The FITS WCS finds pixels from a sky position alone; the gWCS also needs the time and raster step
target = [None, SkyCoord(-320 * u.arcsec, 400 * u.arcsec, frame=iris_frame)]
mg_ii_cut = mg_ii.crop(target, target, wcs=mg_ii.fits_wcs)

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
