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
# If we want to "raster" over wavelength, we can do the following:

fig = plt.figure()
# This will also "transpose" the data but this is only for visualization purposes
# We have to set the vmin and vmax, as by default "plot" works out the
# vmin,vmax from the first slice which in this case is 0.
# By default, the raster scan axis is shown as helioprojective longitude.
mg_ii.plot(fig=fig, plot_axes=["x", "y", None], vmin=0, vmax=1000)

###############################################################################
# If you want to view the raster scan as time instead, select time explicitly.

fig = plt.figure()
mg_ii.plot(
    fig=fig,
    plot_axes=["x", "y", None],
    axes_coordinates=["time", "custom:pos.helioprojective.lat", None],
    vmin=0,
    vmax=1000,
)

###############################################################################
# This object is sliceable, so we can do things like this:

print(mg_ii[120, 200])

fig = plt.figure()
ax = fig.add_subplot(111, projection=mg_ii[120, 200].wcs)
# This is just the data values along the wavelength axis of the Mg II k window at pixel (120, 200)
mg_ii[120, 200].plot(axes=ax)

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
# For raster data, the full gWCS uses ``(spectral, sky, time, scan_step)``
# world inputs, so wavelength-plus-sky queries go through ``basic_wcs``.
# For example, what is the wavelength position that corresponds to
# Mg II k core (279.63 nm)?

iris_observer = wcs_to_celestial_frame(mg_ii.basic_wcs.celestial).observer
iris_frame = Helioprojective(observer=iris_observer)
step_index = mg_ii.data.shape[0] // 2
slit_index = mg_ii.data.shape[1] // 2
_, center_sky = mg_ii.basic_wcs.array_index_to_world(step_index, slit_index, 0)
wcs_loc = mg_ii.basic_wcs.world_to_pixel(
    SpectralCoord(279.63, unit=u.nm),
    center_sky,
)
mg_index = int(np.round(wcs_loc[0]))
print(mg_index)

###############################################################################
# Now we will plot a spectroheliogram for the Mg II k core wavelength.
# As the raster uses a 4D gWCS, we need to provide the full world-object order
# ``(spectral, sky, time, scan_step)``. Here we only constrain wavelength.
lower_corner = [SpectralCoord(280, unit=u.nm), None, None, None]
upper_corner = [SpectralCoord(280, unit=u.nm), None, None, None]
mg_spec_crop = mg_ii.crop(lower_corner, upper_corner)

fig = plt.figure()
ax = fig.add_subplot(111, projection=mg_spec_crop.basic_wcs)
ax.imshow(mg_spec_crop.data, origin="lower", aspect="auto")
ax.coords[0].set_format_unit(u.arcsec)
ax.coords[0].set_axislabel("Helioprojective Latitude [arcsec]")
ax.coords[1].set_format_unit(u.arcsec)
ax.coords[1].set_axislabel("Helioprojective Longitude [arcsec]")

###############################################################################
# Plotting against ``basic_wcs`` shows the sky coordinates directly.
# If you want to view the raster scan as time versus slit position instead,
# plot against the full gWCS via ``mg_spec_crop.plot(...)``.
#
#
# Imagine there's a really cool feature at (-338", 275"), how can you plot
# the spectrum at that location? First locate the nearest raster pixel, then
# convert that pixel back into the full world tuples expected by ``crop``.

target = SkyCoord(-338 * u.arcsec, 275 * u.arcsec, frame=iris_frame)
lon = mg_ii.axis_world_coords_values("custom:pos.helioprojective.lon")[0].to_value(u.arcsec)
lat = mg_ii.axis_world_coords_values("custom:pos.helioprojective.lat")[0].to_value(u.arcsec)
distance = (lon - target.Tx.to_value(u.arcsec)) ** 2 + (lat - target.Ty.to_value(u.arcsec)) ** 2
step_index, slit_index = np.unravel_index(np.nanargmin(distance), distance.shape)
lower_corner = mg_ii.wcs.array_index_to_world(step_index, slit_index, 0)
upper_corner = mg_ii.wcs.array_index_to_world(step_index, slit_index, mg_ii.data.shape[-1] - 1)
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
