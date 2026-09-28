"""
==========================
Produce Mg II Dopplergrams
==========================

In this example, we are going to produce a Dopplergram for the Mg II k line from a
400-step raster. The Dopplergram is obtained by subtracting the intensities at
symmetrical velocity shifts from the line core (e.g., ±50 km/s).
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch
from scipy.interpolate import make_interp_spline

import astropy.units as u
from astropy import constants
from astropy.coordinates import SpectralCoord

from irispy.io import read_files
from irispy.utils import image_clipping

###############################################################################
# `We start with getting data from the IRIS data archive <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20140708_114109_3824262996_2014-07-08T11%3A41%3A092014-07-08T11%3A41%3A09.xml>`__.
# The full observation is available as a `Level 2 raster tarball <https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2014/07/08/20140708_114109_3824262996/iris_l2_20140708_114109_3824262996_raster.tar.gz>`__.
# To keep the download small, we use a cutout of it that only has the Mg II k 2796 window.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.
#
# You will need to update the path to the data in the next section if you do that.

iris_raster_tar = pooch.retrieve(
    "https://github.com/LM-SAL/irispy-data/releases/download/v1/iris_l2_20140708_114109_3824262996_cutout_raster.tar.gz",
    known_hash="8ac7efd70404bdb4fc92794e7694e6248dbe4a6369af82a9fcca697a3bad6015",
)

###############################################################################
# We will now open the data using a helper function which is designed to read
# all files from a single observation.
#
# Since this is a large dataset, we will use memory mapping to read the data values
# directly from the FITS files without loading them into memory.

raster = read_files(iris_raster_tar, memmap=True, spectral_windows="Mg II k 2796")

###############################################################################
# We are after the Mg II k window, which we can select using a key.

mg_ii = raster["Mg II k 2796"]
(mg_wave,) = mg_ii.axis_world_coords("wl")

###############################################################################
# We will plot the spatially averaged spectrum:

plt.figure()
plt.plot(mg_wave.to("nm"), mg_ii.data.mean((0, 1)))
plt.ylabel("DN (Memory Mapped Value)")
plt.xlabel("Wavelength (nm)")

###############################################################################
# This very large dense raster took more than three hours to complete
# across the 400 raster steps (with 30 s exposures), which means that the
# spacecraft's orbital velocity changes during the observations.
# This means that any calibration will need to correct for those shifts.
#
# To illustrate the correction, we will compare the intensity near the
# Mn I line at around 280.2 nm, between the Mg II k and h lines.
#
# For this dataset, the line core of this line falls around 280.2 nm.
# We crop in wavelength space.

lower_corner = [SpectralCoord(280.2, unit=u.nm), None, None, None]
upper_corner = [SpectralCoord(280.2, unit=u.nm), None, None, None]
mg_crop = mg_ii.crop(lower_corner, upper_corner)
# Save the on-disk part of the slit (roughly its first 600 pixels) before modifying
# the data, masking the raw fill value (-32768). We will compare this below.
before = mg_crop.data[:, :600].astype(float)
before[before == -32768] = np.nan
vmin, vmax = np.nanpercentile(before, [1, 99])

###############################################################################
# Orbital motion changes the wavelength sampled at each raster step.
# The radial velocities are stored in the auxiliary data
# extension past the "last" window in the FITS file (column ``OBS_VRIX``).
# ``read_files`` already reads that extension and stores the values, one per
# raster step, in the cube metadata.

v_obs = mg_ii.meta["observer radial velocity"].to("km/s")

plt.figure()
plt.plot(v_obs)
plt.ylabel("Orbital velocity (km/s)")
plt.xlabel("Raster step")

###############################################################################
# To look at intensities at any given raster step we only need to subtract this
# velocity shift from the wavelength scale, but to look at the whole image
# at a given wavelength we must interpolate the original data to take this
# shift into account. Here is a way to do it (note that array dimensions
# apply to this specific example).

c = constants.c.to("km/s")
mn_i_wavelength = 280.2 * u.nm
wave_shift = -v_obs * mn_i_wavelength / c
# Linear interpolation in wavelength, for each raster step
for i in range(mg_ii.data.shape[0]):
    shifted_data = make_interp_spline(
        (mg_wave - wave_shift[i]).to_value(u.nm),
        mg_ii.data[i, :, :],
        k=1,
        axis=-1,
    )(mg_wave.to_value(u.nm), extrapolate=False)
    mg_ii.data[i, :, :] = np.nan_to_num(shifted_data, nan=0.0)

###############################################################################
# In this observation the correction shifts the spectra by less than one wavelength
# pixel, so the solar structures look very similar before and after correction.
# We use the same intensity scale for both images and show their difference on a
# separate scale centred on zero to make the small changes visible.

# Since we changed the underlying data, we need to re-crop
mg_crop = mg_ii.crop(lower_corner, upper_corner)
after = mg_crop.data[:, :600].astype(float)
after[after == -32768] = np.nan
difference = after - before
limit = np.nanpercentile(np.abs(difference), 99)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True, layout="constrained")
for ax, data, title in zip(axes[:2], (before, after), ("Before correction", "After correction"), strict=True):
    ax.imshow(data, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)

change = axes[2].imshow(difference, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
axes[2].set_title("After minus before")
fig.colorbar(change, ax=axes[2], label="Intensity change (DN)")
for ax in axes:
    ax.set_xlabel("Position along the slit (pixel)")
axes[0].set_ylabel("Raster step")

###############################################################################
# This correction accounts for the measured orbital velocity; other wavelength
# shifts can remain. A more elaborate correction can be obtained by the IDL routine
# ``iris_prep_wavecorr_l2``, but this has not yet been ported to Python
# see the `IDL version of this
# tutorial <http://iris.lmsal.com/itn26/tutorials.html#mg-ii-dopplergrams>`__
# for more details.
#
# We can use the calibrated data for example to calculate Dopplergrams. A
# Dopplergram is here defined as the difference between the intensities at
# two wavelength positions at the same (and opposite) distance from the
# line core. For example, at +/- 50 km/s from the Mg II k3 core. To do
# this, let us first calculate a velocity scale for the k line and find
# the indices of the -50 and +50 km/s velocity positions (here using the
# convention of negative velocities for up flows):

mg_k_centre = 279.6351 * u.nm
pos = 50 * u.km / u.s  # Around the line centre
velocity = (mg_wave - mg_k_centre) * c / mg_k_centre
index_p = np.argmin(np.abs(velocity - pos))
index_m = np.argmin(np.abs(velocity + pos))
doppler = mg_ii.data[..., index_m] - mg_ii.data[..., index_p]

###############################################################################
# And now we can plot this as before (intensity units are again arbitrary
# because of the unscaled DNs).

vmin, vmax = image_clipping(doppler)
# A diverging colour map needs limits centred on zero.
limit = max(abs(vmin), abs(vmax))
plt.figure()
plt.imshow(
    doppler.T,
    cmap="RdBu",
    origin="lower",
    aspect=0.5,
    vmin=-limit,
    vmax=limit,
)
plt.colorbar()
# This plots the array itself, so the axes are array indices rather than solar coordinates.
plt.xlabel("Raster step")
plt.ylabel("Position along the slit (pixel)")
plt.tight_layout()

plt.show()

# sphinx_gallery_thumbnail_number = 4
