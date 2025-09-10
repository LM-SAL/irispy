"""
=============================
Dealing with IRIS v34 rasters
=============================

In this example we will show how ``irispy`` deals with a v34 dataset by default and how to
undo that if you so desire.

These v34 are scans which raster from west to east instead of the default east to west.
"""

import matplotlib.pyplot as plt
import pooch

import astropy.units as u
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.wcs.utils import wcs_to_celestial_frame

from sunpy.coordinates.frames import Helioprojective

from irispy.io import read_files

###############################################################################
# We start by downloading a raster v34 dataset from the IRIS archive.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but using your browser will also work.
#
# This is a very large sparse 64-step raster with a FOV of 63"x175" for
# a total of 8 complete raster scans.

raster_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2025/03/28/20250328_225628_3400109360/iris_l2_20250328_225628_3400109360_raster.tar.gz",
    known_hash="ce46e93a8962b04da344c203c4f75a8d7411d1f587e25d728a6ca8398208de1a",
)

###############################################################################
# We will now open the raster file we just downloaded.
# By default, irispy will read the v34 data, flipping the data so that it
# is in the same orientation as normal IRIS data and adjust the WCS accordingly.

raster = read_files(raster_filename, memmap=False)
# We will also undo the v34 handling and read the data as is.
raster_unflipped = read_files(raster_filename, memmap=False, revert_v34=True)

# Printing will give us an overview of the file.
print(raster)

###############################################################################
# We will now pull the 'Mg II k 2796' spectral line data from the raster.

mg_ii_k = raster["Mg II k 2796"]
mg_ii_k_unflipped = raster_unflipped["Mg II k 2796"]
print(mg_ii_k)

###############################################################################
# To see the effect of the v34 handling, we will plot a spectroheliogram
# for the Mg II k core wavelength.
#
# We can use the ``crop`` method to get this information, this will
# require a `astropy.coordinates.SpectralCoord` object from `astropy.coordinates`.

# None, means that the axis is not cropped
# Since we want one physical coordinate, we will just use the
# same spectral coordinate for both axes.
lower_corner = [SpectralCoord(280, unit=u.nm), None]

mg_spec_crop = mg_ii_k[0].crop(lower_corner, lower_corner)
mg_spec_unflipped_crop = mg_ii_k_unflipped[0].crop(lower_corner, lower_corner)

fig = plt.figure(figsize=(6, 12))
ax = fig.add_subplot(121, projection=mg_spec_crop.wcs)
ax.set_title("v34 flipped")
mg_spec_crop.plot(axes=ax, plot_axes=["x", "y"])

ax2 = fig.add_subplot(122, projection=mg_spec_unflipped_crop.wcs)
ax2.set_title("v34 unflipped")
mg_spec_unflipped_crop.plot(axes=ax2, plot_axes=["x", "y"])
fig.tight_layout()

###############################################################################
# As you can see, the v34 data is flipped in the y-axis and the WCS is
# adjusted accordingly.
#
# The same is true for the times in the raster:

print(f"Flipped time: {mg_ii_k.time[:5]}")
print("*" * 50)
print(f"Unflipped time: {mg_ii_k_unflipped.time[:5]}")

###############################################################################
# Finally we will just see that the spectral profiles are unaffected in either case.

iris_observer = wcs_to_celestial_frame(mg_ii_k[0].wcs.celestial).observer
iris_frame = Helioprojective(observer=iris_observer)
lower_corner = [None, SkyCoord(-909 * u.arcsec, 294 * u.arcsec, frame=iris_frame)]
mg_ii_k_unflipped_spectra = mg_ii_k_unflipped[0].crop(lower_corner, lower_corner)
mg_ii_k_spectra = mg_ii_k[0].crop(lower_corner, lower_corner)

fig = plt.figure()
ax = fig.add_subplot(111, projection=mg_ii_k_unflipped_spectra.wcs)
mg_ii_k_unflipped_spectra.plot(axes=ax, color="red", label="v34 unflipped")
mg_ii_k_spectra.plot(axes=ax, color="black", label="v34 default", linestyle="--")
plt.legend()

###############################################################################

plt.show()
