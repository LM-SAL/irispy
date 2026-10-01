"""
==========================
Produce Mg II Dopplergrams
==========================

In this example, we are going to produce a Dopplergram for the Mg II k line from a
400-step raster. The Dopplergram is obtained by subtracting the intensities at
symmetrical velocity shifts from the line core (e.g., ±50 km/s), so every step needs
the same wavelength scale. We measure and correct its drift with
`~irispy.utils.wavelength_drift.calculate_wavelength_drift`, as the
`IDL tutorial <https://iris.lmsal.com/itn26/tutorials.html#mg-ii-dopplergrams>`__ does
with ``iris_prep_wavecorr_l2``.
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
from irispy.utils.wavelength_drift import calculate_wavelength_drift

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

mg_ii = raster["Mg II k 2796"][0]
(mg_wave,) = mg_ii.axis_world_coords("wl")

###############################################################################
# We will plot the spatially averaged spectrum:

plt.figure()
plt.plot(mg_wave.to("nm"), mg_ii.data.mean((0, 1)))
plt.ylabel("DN (Memory Mapped Value)")
plt.xlabel("Wavelength (nm)")

###############################################################################
# This very large dense raster took more than three hours to complete
# across the 400 raster steps (with 30 s exposures). Over that time the
# spacecraft's orbital velocity and the temperature of the spectrograph
# change, and both move the spectra in wavelength.
#
# To see the problem, let us look at how the line intensity varies for a
# strong Mn I line at around 280.2 nm, in between the Mg II k and h lines.
# We crop in wavelength space.

lower_corner = [SpectralCoord(280.2, unit=u.nm), None]
upper_corner = [SpectralCoord(280.2, unit=u.nm), None]
mg_crop = mg_ii.crop(lower_corner, upper_corner)
# The raw values include fill values (-32768) and the dark sky above the limb, so we take
# the colour limits from the part of the slit on the disk (roughly its first 600 pixels).
vmin, vmax = np.percentile(mg_crop.data[:, :600], [1, 99])
plt.figure()
# We will "crunch" the image a bit using the aspect ratio.
mg_crop.plot(aspect="auto", vmin=vmin, vmax=vmax)

###############################################################################
# You can see a regular bright-dark pattern along the y-axis (the raster steps), an
# indication that the intensities are not taken at the same position in
# the line because of wavelength shifts.
#
# The shifts can be measured from photospheric lines of known rest wavelength:
# this window holds the Ni I 279.9474, Mn I 280.1902 and Fe I 280.5346 nm
# absorption lines. `~irispy.utils.wavelength_drift.calculate_wavelength_drift`
# fits each of them in the spectrum averaged along the slit at every step, then
# fits a sine with the orbital period, plus a slow polynomial, to the Ni I shifts.
# It reads only the fit ranges, so the memory-mapped data are fine. The table
# has one row per step.

drift = calculate_wavelength_drift(raster)
drift[:5]

###############################################################################
# Let us plot the measured shifts and the fitted NUV drift against time, with
# the Doppler shift of the spacecraft's orbital velocity alone, which the Level
# 2 files record in their auxiliary data. The difference between the two curves
# is the thermal drift of the spectrograph.

minutes = (drift["time"] - drift["time"][0]).to_value(u.min)
v_obs = mg_ii.meta["observer radial velocity"]
orbital = (v_obs / constants.c * 279.9474 * u.nm).to_value(u.AA)
plt.figure()
for name in ("Ni I", "Mn I", "Fe I"):
    plt.plot(minutes, drift[name].to_value(u.AA), ".", label=name)
plt.plot(minutes, drift["nuv"].to_value(u.AA), "k", label="NUV drift")
plt.plot(minutes, orbital, "k--", label="Orbital velocity alone")
plt.xlabel("Time since the first step (min)")
plt.ylabel("Wavelength shift (Å)")
plt.legend()

###############################################################################
# Adding the drift to the wavelengths of a step corrects them. To look at the
# whole image at a given wavelength, we instead interpolate each step back onto
# the original wavelength grid.

for i, shift in enumerate(drift["nuv"]):
    mg_ii.data[i] = make_interp_spline((mg_wave + shift).to_value(u.nm), mg_ii.data[i], k=1, axis=-1)(
        mg_wave.to_value(u.nm)
    )

###############################################################################
# Now we can plot the shifted data to see that the large scale shifts
# have disappeared

plt.figure()
# Since we changed the underlying data, we need to re-crop
mg_crop = mg_ii.crop(lower_corner, upper_corner)
# We will "crunch" the image a bit using the aspect ratio, with the same colour limits as before.
mg_crop.plot(aspect="auto", vmin=vmin, vmax=vmax)

###############################################################################
# We can use the corrected data for example to calculate Dopplergrams. A
# Dopplergram is here defined as the difference between the intensities at
# two wavelength positions at the same (and opposite) distance from the
# line core. For example, at +/- 50 km/s from the Mg II k3 core. To do
# this, let us first calculate a velocity scale for the k line and find
# the indices of the -50 and +50 km/s velocity positions (here using the
# convention of negative velocities for up flows):

mg_k_centre = 279.6351 * u.nm
pos = 50 * u.km / u.s  # Around the line centre
velocity = ((mg_wave - mg_k_centre) * constants.c / mg_k_centre).to(u.km / u.s)
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
