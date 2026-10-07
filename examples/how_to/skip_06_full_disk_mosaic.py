"""
=======================
Read a full-disk mosaic
=======================

In this example we read the Mg II k full-disk mosaic of 2015-02-22 and compare a map at
the center of the window with a map of its blue end.

The file is 553 MB, so this example is not run when the documentation is built.
"""

import matplotlib.pyplot as plt
import pooch

import astropy.units as u

from irispy.io import read_mosaic

###############################################################################
# We start with getting data from the `IRIS mosaic page <https://iris.lmsal.com/mosaic.html>`__.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.

mosaic_file = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2015/02/22/20150222Mosaic/IRISMosaic_20150222_MgIIk.fits.gz",
    known_hash="dde33eb7a667817fab8e206f07a05b369bfe10fcca3efa64b4efd6562ee5754d",
)

###############################################################################
# The whole cube takes 2.4 GB in memory, so we read only the blue half of the window.

mosaic = read_mosaic(mosaic_file, wavelength_range=[2794.6, 2796.4] * u.AA)
print(mosaic)

###############################################################################
# ``meta["time"]`` holds the time of each position, masked where no raster covered it.

times = mosaic.meta["time"]
print(times.min().isot, times.max().isot)

###############################################################################
# `~irispy.spectrograph.MosaicCube.to_map` gives a `sunpy.map.Map` at the nearest
# wavelength (here ``LAMREF``, the window center) or the mean over a range (here the
# bluest 0.4 Å).

core = mosaic.to_map(mosaic.meta.rest_wavelength)
wing = mosaic.to_map([2794.6, 2795.0] * u.AA)

###############################################################################
# The mosaic pixels are 2 arcsec wide and 1/3 arcsec tall, so we set the pixel aspect
# ratio when plotting.

aspect = (core.scale.axis2 / core.scale.axis1).value
fig = plt.figure(figsize=(12, 6))
ax_core = fig.add_subplot(121, projection=core)
core.plot(axes=ax_core, clip_interval=(1, 99.5) * u.percent, aspect=aspect)
ax_core.set_title(f"Mg II k, {core.wavelength:.2f}")
ax_wing = fig.add_subplot(122, projection=wing)
wing.plot(axes=ax_wing, clip_interval=(1, 99.5) * u.percent, aspect=aspect)
ax_wing.set_title("Mg II k window, 2794.6-2795.0 Å")
fig.tight_layout()

plt.show()
