"""
==============================
Fit Spectral Models to Spectra
==============================

In this example, we are going to fit Si IV 1403 from IRIS with a single Gaussian, starting
from `~irispy.utils.fitting.si_iv_1403_model`. Then we will use the fitted values to make
maps of the line's flux, Doppler shift, width and non-thermal velocity.

:ref:`irispy-tutorial-fitting` explains the fitting call and what to watch out for.
For a model-independent alternative, the spectral moments, see
:ref:`sphx_glr_generated_gallery_analysis_04_spectral_moments.py`.

If you want to see a similar example but with a double Gaussian fit to the Mg II k line,
see :ref:`sphx_glr_generated_gallery_analysis_07_mg_ii_two_gaussian_fitting.py`.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch

import astropy.units as u
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.modeling.fitting import TRFLSQFitter, parallel_fit_dask
from astropy.wcs.utils import wcs_to_celestial_frame

from sunpy.coordinates.frames import Helioprojective

from irispy.io import read_files
from irispy.utils.fitting import maps_from_fit, non_thermal_velocity, si_iv_1403_model

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
# We read only the Si IV 1403 window, with its uncertainties to weight the fit,
# and select the one complete scan.

raster = read_files(raster_filename, spectral_windows="Si IV 1403", uncertainty=True)
si_iv_1403 = raster["Si IV 1403"][0]

###############################################################################
# Before we get to fitting, we will shrink the data cube to make it easier to work with.
# This is done primarily to speed up the fitting process on the online documentation build.

iris_observer = wcs_to_celestial_frame(si_iv_1403.wcs.celestial).observer
iris_frame = Helioprojective(observer=iris_observer)
top_left = [None, SkyCoord(-360 * u.arcsec, 310 * u.arcsec, frame=iris_frame)]
bottom_right = [None, SkyCoord(-290 * u.arcsec, 260 * u.arcsec, frame=iris_frame)]
si_iv_1403 = si_iv_1403.crop(top_left, bottom_right)
# We also average 2x2 spatial pixels, after trimming both spatial axes to an even length,
# keeping the uncertainties. This improves the signal-to-noise of the faint Si IV line
# and means 4x fewer fits.
ny, nx = (n // 2 * 2 for n in si_iv_1403.data.shape[:2])
si_iv_1403 = si_iv_1403[:ny, :nx].rebin((2, 2, 1), propagate_uncertainties=True)

###############################################################################
# The model describes Si IV alone, so we keep the wavelengths halfway to the
# neighboring O IV lines at 1401.157 and 1404.806 Å :cite:p:`depontieu2014,polito2016`.

blue, si_iv_core, red = [1401.157, 1402.77, 1404.806] * u.AA
lower_corner = [SpectralCoord((blue + si_iv_core) / 2), None]
upper_corner = [SpectralCoord((si_iv_core + red) / 2), None]
si_iv_1403 = si_iv_1403.crop(lower_corner, upper_corner)

###############################################################################
# Let us just get the full field of view at the line core.

si_iv_spec_crop = si_iv_1403.crop([SpectralCoord(si_iv_core), None], [SpectralCoord(si_iv_core), None])

###############################################################################
# `~irispy.utils.fitting.si_iv_1403_model` starts every spectrum's Gaussian from the
# data, on a constant background. We fit it to every spectrum with
# `~astropy.modeling.fitting.parallel_fit_dask`, passing the wavelengths in Å and
# giving the masked samples no weight.

model = si_iv_1403_model(si_iv_1403)
wavelength = si_iv_1403.axis_world_coords("em.wl")[0].to(u.AA)
good = np.isfinite(si_iv_1403.data) & ~si_iv_1403.mask
fitter = TRFLSQFitter()
iris_model_fit = parallel_fit_dask(
    model=model,
    fitter=fitter,
    data=np.where(good, si_iv_1403.data, 0),
    data_unit=si_iv_1403.unit,
    weights=np.where(good, 1 / si_iv_1403.uncertainty.array, 0),
    world=(wavelength,),
    fitting_axes=2,
    fit_info=True,
    scheduler="single-threaded",
)

###############################################################################
# `~irispy.utils.fitting.maps_from_fit` turns the fitted parameters into maps with the
# spatial coordinates of the cube and uncertainties from the fit. It also derives
# the Doppler velocity, width and integrated intensity against the Si IV rest wavelength.

maps = maps_from_fit(iris_model_fit, si_iv_1403, fitter=fitter)
print(list(maps.keys()))

###############################################################################
# We compare the starting and fitted models for a bright spectrum, the one at the
# 99th percentile of integrated intensity.

intensity = maps["integrated_intensity_1"].data
# The models' parameters are maps, so we evaluate them on wavelengths shaped to broadcast against them.
bright = np.nanpercentile(intensity, 99)
step, slit = np.unravel_index(np.nanargmin(np.abs(intensity - bright)), intensity.shape)
plt.figure()
ax = si_iv_1403[step, slit].plot(label="Spectrum")
ax.plot(model(wavelength[:, np.newaxis, np.newaxis])[:, step, slit], label="Starting model")
ax.plot(iris_model_fit(wavelength[:, np.newaxis, np.newaxis])[:, step, slit], linestyle="--", label="Fitted model")
ax.set_title("Si IV 1403 profile")
plt.legend()

###############################################################################
# Now we plot them next to the line-core image.

fig, ax_dict = plt.subplot_mosaic(
    [["fov", "intensity"], ["velocity", "width"]],
    subplot_kw={"projection": si_iv_spec_crop.wcs},
    figsize=(12, 10),
)

si_iv_spec_crop.plot(axes=ax_dict["fov"], plot_axes=["x", "y"], vmin=0, vmax=200)
ax_dict["fov"].set_title(f"Si IV {si_iv_core.to_value(u.AA)} Å")
fig.colorbar(ax_dict["fov"].images[0], ax=ax_dict["fov"], label="Intensity [DN]", shrink=0.8)

for key, name, label, symmetric in [
    ("intensity", "integrated_intensity_1", "Integrated intensity", False),
    ("velocity", "velocity_1", "Doppler shift", True),
    ("width", "fwhm_velocity_1", "Line width (FWHM)", False),
]:
    limit = np.nanpercentile(np.abs(maps[name].data), 95)
    maps[name].plot(
        axes=ax_dict[key],
        plot_axes=["x", "y"],
        cmap="coolwarm" if symmetric else None,
        vmin=-limit if symmetric else 0,
        vmax=limit,
    )
    cbar = fig.colorbar(ax_dict[key].images[0], ax=ax_dict[key], extend="both" if symmetric else "max")
    cbar.set_label(label=f"{label} [{maps[name].unit.to_string()}]", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    ax_dict[key].set_title(label)


def label_axes(ax):
    # The first world axis is latitude, along the slit (y), and the second is longitude, along the raster (x).
    for coord, side in ((ax.coords[0], "l"), (ax.coords[1], "b")):
        coord.set_ticklabel(exclude_overlapping=True, fontsize=8)
        coord.set_ticks_position(side)
        coord.set_ticklabel_position(side)
        coord.set_axislabel_position(side)


for ax in ax_dict.values():
    label_axes(ax)
fig.tight_layout()

###############################################################################
# The fitted width also holds the instrumental and thermal broadening of the line.
# `~irispy.utils.fitting.non_thermal_velocity` removes both, taking the thermal width
# at the temperature we give, here the peak of Si IV in CHIANTI's ionisation
# equilibrium, log T = 4.9 (Dere et al. 2023). Lines narrower than that have no
# non-thermal velocity.

non_thermal = non_thermal_velocity(maps["fwhm_1"], si_iv_core, ion="Si IV", temperature=10**4.9 * u.K)[
    "non_thermal_velocity"
]

fig = plt.figure()
ax = fig.add_subplot(projection=non_thermal.wcs)
non_thermal.plot(axes=ax, plot_axes=["x", "y"], vmin=0, vmax=np.nanpercentile(non_thermal.data, 95))
fig.colorbar(ax.images[0], ax=ax, extend="max", label=f"Non-thermal velocity [{non_thermal.unit.to_string()}]")
ax.set_title("Si IV non-thermal velocity")
label_axes(ax)

plt.show()

# sphinx_gallery_thumbnail_number = 2
