"""
===============================
Working with spectrograph files
===============================

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

quantity_support()

###############################################################################
# We start with getting data from the IRIS archive.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but using your browser will also work.

raster_filename = pooch.retrieve(
    "http://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2018/01/02/20180102_153155_3610108077/iris_l2_20180102_153155_3610108077_raster.tar.gz",
    known_hash="8949562149cfa5fba067b5b102e8434b14cea3c3416dd79c06b7f6e211c61a39",
)

###############################################################################
# Note that when ``memmap=True``, the data values are read from the FITS file
# directly without the scaling to Float32, the data values are no longer in DN,
# but in scaled integer units that start at -2$^{16}$/2.

raster = read_files(raster_filename, memmap=False)

###############################################################################
# Let us now explore what was returned.

# Provides an overview of the Spectrograph object
print(raster)

# Will give us all the keys that corresponds to all the wavelength windows.
print(raster.keys())

# We can get the Mg II k window
mg_ii = raster["Mg II k 2796"]
print(mg_ii)

###############################################################################
# This is a `irispy.spectrograph.SpectrogramCubeSequence` which contains each
# complete raster as one individual `irispy.spectrograph.SpectrogramCube` object.
# In this case, it was only one complete raster, so the first axis is only length 1.
#
# So we will index to get the first raster and work with that.

mg_ii = mg_ii[0]
print(mg_ii)

###############################################################################
# Now we have more information about the data, including the OBS ID and description.
#
# Let's plot it

fig = plt.figure()
mg_ii.plot(fig=fig)

###############################################################################
# If we want to "raster" over wavelength, we can do the following

fig = plt.figure()
# This will also "transpose" the data but this is only for visualization purposes
# We have to set the vmin and vmax, as by default "plot" works out the
# vmin,vmax from the first slice which in this case is 0.
mg_ii.plot(fig=fig, plot_axes=["x", "y", None], vmin=0, vmax=1000)

###############################################################################
# This object is sliceable, so we can do things like this:

print(mg_ii[120, 200])

fig = plt.figure()
ax = fig.add_subplot(111, projection=mg_ii[120, 200].wcs)
# This is just the data values along the wavelength axis of the Mg II k window at pixel (120, 200)
mg_ii[120, 200].plot(axes=ax)

###############################################################################
# We can also plot using the data directly. We can read the wavelengths of the
# Mg window by calling `ndcube.NDCube.axis_world_coords` for "wl" (wavelength), and redo the plot.

(mg_wave,) = mg_ii.axis_world_coords("wl")

fig, ax = plt.subplots()
ax.plot(mg_wave.to("AA"), mg_ii.data[120, 200])

###############################################################################
# When we use the underlying data directly, we lose all the metadata and WCS information.
#
# If you are unfamiliar with WCS, the following links are quite useful:
#
# * https://docs.astropy.org/en/stable/wcs/index.html
# * https://docs.astropy.org/en/stable/visualization/wcsaxes/index.html
#
# Some of the higher-level utilities are via ndcube, e.g., coordinate transformations: https://docs.sunpy.org/projects/ndcube/en/stable/explaining_ndcube/coordinates.html.
#
# Now, let's take a look at the WCS information.
# For example, what is the wavelength position that corresponds to Mg II k core (279.63 nm)?

iris_observer = wcs_to_celestial_frame(mg_ii.wcs.celestial).observer
iris_frame = Helioprojective(observer=iris_observer)
wcs_loc = mg_ii.wcs.world_to_pixel(
    SpectralCoord(279.63, unit=u.nm),
    SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=iris_frame),
)
mg_index = int(np.round(wcs_loc[0]))
print(mg_index)

###############################################################################
# Now we will plot spectroheliogram for Mg II k core wavelength.
# We can use the ``crop`` method to get this information, this will
# require a `astropy.coordinates.SpectralCoord` object from `astropy.coordinates`.

# None, means that the axis is not cropped
# Note that this has to be in axis order
lower_corner = [SpectralCoord(280, unit=u.nm), None]
upper_corner = [SpectralCoord(280, unit=u.nm), None]
mg_spec_crop = mg_ii.crop(lower_corner, upper_corner)

fig = plt.figure()
ax = fig.add_subplot(111, projection=mg_spec_crop.wcs)
mg_spec_crop.plot(axes=ax)

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
#  Now, you may also be interested in knowing the time that was this observation taken.
# There is some information in ``.meta``.

print(mg_ii.meta)

###############################################################################
# But this is mostly about the observation in general.
# Times of individual scans are saved in .extra_coords['time'].
# Getting access to it can be done in the following  way:

print(mg_ii.axis_world_coords("time", wcs=mg_ii.extra_coords))
