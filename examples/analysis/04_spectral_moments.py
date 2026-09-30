"""
===============================
Calculate Spectral Line Moments
===============================

In this example, we are going to calculate the spectral moments from IRIS raster data.
Moments provide a model-independent way to characterize spectral lines:

* 0th moment gives the total intensity
* 1st moment gives the centroid (Doppler shift)
* 2nd moment gives the line width

This is in direct contrast to fitting a model to the data which is done in example
:ref:`sphx_glr_generated_gallery_analysis_01_spectral_fitting.py` where we fit a Gaussian to the
line profile and extract the same information from the fit parameters.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch

import astropy.units as u
from astropy.coordinates import SkyCoord, SpectralCoord

from irispy.io import read_files
from irispy.utils.moments import calculate_moments

###############################################################################
# `We start with getting data from the IRIS data archive <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20180102_153155_3610108077_2018-01-02T15%3A31%3A552018-01-02T15%3A31%3A55.xml>`__.
# The full observation is available as a `Level 2 raster tarball <https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2018/01/02/20180102_153155_3610108077/iris_l2_20180102_153155_3610108077_raster.tar.gz>`__.
# To keep the download small, we use a cutout of it that only has the Si IV 1403 and Mg II k 2796 windows.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.
#
# You will need to update the path to the data in the next section if you do that.

raster_filename = pooch.retrieve(
    "https://github.com/LM-SAL/irispy-data/releases/download/v1/iris_l2_20180102_153155_3610108077_cutout_raster.tar.gz",
    known_hash="ff80e6a7900d4d5e1716a6415db25d40b6058f3523184b549c7e0d9928c0b68b",
)

###############################################################################
# We will now open the data using a helper function which is designed to read
# all files from a single observation.

raster = read_files(raster_filename, spectral_windows="Si IV 1403")

###############################################################################
# We will just focus on the Si IV 1403 line which we can select using a key.

si_iv_1403 = raster["Si IV 1403"]

# However, before we get to that, we will shrink the data cube to make it easier to work with.
iris_frame = si_iv_1403.celestial_frame
top_left = SkyCoord(-360 * u.arcsec, 310 * u.arcsec, frame=iris_frame)
bottom_right = SkyCoord(-290 * u.arcsec, 260 * u.arcsec, frame=iris_frame)
crop_wavelength = SpectralCoord(140.277, unit=u.nm)
bottom_step, bottom_slit_pixel, _ = si_iv_1403.fits_wcs.world_to_array_index(crop_wavelength, bottom_right)
top_step, top_slit_pixel, _ = si_iv_1403.fits_wcs.world_to_array_index(crop_wavelength, top_left)
si_iv_1403 = si_iv_1403.crop(
    si_iv_1403.wcs.array_index_to_world(bottom_step, bottom_slit_pixel, 0),
    si_iv_1403.wcs.array_index_to_world(top_step, top_slit_pixel, si_iv_1403.data.shape[-1] - 1),
)

###############################################################################
# Let us just check the full field of view at the line core.

si_iv_core = 140.277 * u.nm
lower_corner = [SpectralCoord(si_iv_core), None, None, None]
upper_corner = [SpectralCoord(si_iv_core), None, None, None]
si_iv_spec_crop = si_iv_1403.crop(lower_corner, upper_corner)

###############################################################################
# Now we can calculate the spectral moments using the `~irispy.utils.moments.calculate_moments` function.
#
# This helper function automatically extracts the wavelength coordinates from the cube's
# WCS and computes the moments along the spectral axis for every spatial pixel.
#
# We will restrict the calculation to a narrow window around the rest wavelength
# (0.05 nm = 0.5 Å on each side) to isolate the Si IV line from its neighbors.
#
# While ``wings`` is not required, it is often a good idea to restrict the
# calculation to a window around the line of interest to avoid contamination
# from other lines or noise in the continuum.
#
# The same goes for ``rest_wavelength``, which is used to calculate the velocity
# from the wavelength shift in the 1st moment, otherwise you get the ``centroid``
# in wavelength units instead of velocity units and the same goes for the line
# width from the 2nd moment.
#
# Where the line is faint, the window is mostly noise and the moments say little about
# the line (noise alone gives a width of about 60 km/s here). With ``min_intensity`` we
# only keep pixels with a total intensity of at least 200 DN, and the others are left blank.

moments = calculate_moments(
    si_iv_1403, rest_wavelength=si_iv_core, wings=0.05 * u.nm, integrated=False, min_intensity=200 * si_iv_1403.unit
)
# The return is a RasterCollection of 2D maps, one for each moment; it also has the
# "centroid" and "width" in wavelength units.
intensity = moments["intensity"]
velocity = moments["velocity"]
velocity_width = moments["velocity_width"]

###############################################################################
# We will now visualize the moments. Note that the output is a
# `~irispy.spectrograph.RasterCollection` which contains 2D
# `~irispy.spectrograph.SpectrogramCube` objects with the spatial WCS preserved
# from the input cube.

fig, ax_dict = plt.subplot_mosaic(
    [["fov", "intensity"], ["velocity", "width"]],
    subplot_kw={"projection": si_iv_spec_crop.wcs},
    figsize=(12, 10),
)

si_iv_spec_crop.plot(axes=ax_dict["fov"], plot_axes=["x", "y"], vmin=0, vmax=200)
ax_dict["fov"].set_title("Si IV 1402.77 Å")
fig.colorbar(ax_dict["fov"].images[0], ax=ax_dict["fov"], label="Intensity [DN]", shrink=0.8)

# 0th moment: Total intensity
amp_max = np.nanpercentile(np.abs(intensity.data), 99)
intensity.plot(axes=ax_dict["intensity"], plot_axes=["x", "y"], vmin=0, vmax=amp_max)
cbar = fig.colorbar(ax_dict["intensity"].images[0], ax=ax_dict["intensity"])
cbar.set_label(label=f"Intensity [{intensity.unit.to_string()}]", fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax_dict["intensity"].set_title("Total Intensity (0th Moment)")

# 1st moment: Doppler velocity from centroid shift
shift_max = np.nanpercentile(np.abs(velocity.data), 95)
velocity.plot(axes=ax_dict["velocity"], plot_axes=["x", "y"], cmap="coolwarm", vmin=-shift_max, vmax=shift_max)
cbar = fig.colorbar(ax_dict["velocity"].images[0], ax=ax_dict["velocity"], extend="both")
cbar.set_label(label=f"Doppler shift [{velocity.unit.to_string()}]", fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax_dict["velocity"].set_title("Velocity from Centroid")

# 2nd moment: Line width, in velocity units
wmax = np.nanpercentile(velocity_width.data, 95)
velocity_width.plot(axes=ax_dict["width"], plot_axes=["x", "y"], vmax=wmax)
cbar = fig.colorbar(ax_dict["width"].images[0], ax=ax_dict["width"])
cbar.set_label(label=f"Width [{velocity_width.unit.to_string()}]", fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax_dict["width"].set_title("Line Width (2nd Moment)")

for ax in ax_dict.values():
    # Longitude and latitude are already on the raster step and slit edges; just shrink their labels.
    for index in (0, 1):
        ax.coords[index].set_ticklabel(exclude_overlapping=True, fontsize=8)
fig.tight_layout()

plt.show()
