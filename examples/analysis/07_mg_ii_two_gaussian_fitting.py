"""
========================================================
Fit Spectral Models to Spectra - Double Gaussian Fitting
========================================================

In this example, we are going to fit the Mg II k line core from IRIS raster data with a double
Gaussian model, starting from `~irispy.utils.fitting.mg_ii_model`. Then we will use the fitted
values to make maps of the total flux, the blue-red flux asymmetry and the separation of the two
components.

:ref:`irispy-tutorial-fitting` explains the fitting call and what to watch out for, and
:ref:`sphx_glr_generated_gallery_analysis_01_spectral_fitting.py` fits a single Gaussian to Si IV 1403.
For a model-independent alternative, the spectral moments, see
:ref:`sphx_glr_generated_gallery_analysis_04_spectral_moments.py`.
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
from irispy.spectrograph import SpectrogramCube
from irispy.utils.fitting import FitQualityFlag, fit_to_maps, mg_ii_model
from irispy.utils.lines import get_lines

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
#
# We read only the Mg II k window, with its uncertainties to weight the fit,
# and select the one complete scan.

raster = read_files(raster_filename, spectral_windows="Mg II k 2796", uncertainty=True)
mg_ii_k = raster["Mg II k 2796"][0]

###############################################################################
# We crop the spatial field of view to keep the example light enough
# for the documentation build. We will also focus on the Mg II k core,
# which is the part of the spectrum we are going to fit: the model has no
# components for the line wings or for the Mg II triplet next to it.

iris_observer = wcs_to_celestial_frame(mg_ii_k.wcs.celestial).observer
iris_frame = Helioprojective(observer=iris_observer)
top_left = [None, SkyCoord(-350 * u.arcsec, 310 * u.arcsec, frame=iris_frame)]
bottom_right = [None, SkyCoord(-290 * u.arcsec, 260 * u.arcsec, frame=iris_frame)]
mg_ii_k = mg_ii_k.crop(top_left, bottom_right)

lower_corner = [SpectralCoord(279.40, unit=u.nm), None]
upper_corner = [SpectralCoord(279.80, unit=u.nm), None]
mg_ii_k = mg_ii_k.crop(lower_corner, upper_corner)
# We also average 2x2 spatial pixels, after trimming both spatial axes to an even length,
# keeping the uncertainties, which means 4x fewer spectra to fit.
ny, nx = (n // 2 * 2 for n in mg_ii_k.data.shape[:2])
mg_ii_k = mg_ii_k[:ny, :nx].rebin((2, 2, 1), propagate_uncertainties=True)

###############################################################################
# `~irispy.utils.fitting.mg_ii_model` starts one Gaussian at the k2v peak and one
# at the k2r peak of every spectrum, where `~irispy.utils.mg_features.calculate_mg_features`
# finds them, on a constant background. We fit it to every spectrum with
# `~astropy.modeling.fitting.parallel_fit_dask`, passing the wavelengths in Å and
# giving the masked samples no weight.

model = mg_ii_model(mg_ii_k)
wavelength = mg_ii_k.axis_world_coords("em.wl")[0].to(u.AA)
good = np.isfinite(mg_ii_k.data) & ~mg_ii_k.mask
fitter = TRFLSQFitter()
mg_ii_model_fit = parallel_fit_dask(
    model=model,
    fitter=fitter,
    data=np.where(good, mg_ii_k.data, 0),
    data_unit=mg_ii_k.unit,
    weights=np.where(good, 1 / mg_ii_k.uncertainty.array, 0),
    world=(wavelength,),
    fitting_axes=2,
    fit_info=True,
    scheduler="single-threaded",
)

###############################################################################
# We compare the starting and the fitted model for one spectrum.

# The models' parameters are maps, so we evaluate them on wavelengths shaped to broadcast against them.
step, slit = (n // 2 for n in model.mean_1.shape)
plt.figure()
ax = mg_ii_k[step, slit].plot(label="Spectrum")
ax.plot(model(wavelength[:, np.newaxis, np.newaxis])[:, step, slit], label="Starting model")
ax.plot(mg_ii_model_fit(wavelength[:, np.newaxis, np.newaxis])[:, step, slit], linestyle="--", label="Fitted model")
ax.set_title("Mg II k profile")
plt.legend()

###############################################################################
# Now we will produce maps of the total fitted flux, the blue-red flux asymmetry of the
# two components, (blue - red) / total, and the separation of their peaks.
#
# These follow the Mg II k diagnostics of :cite:t:`leenaarts2013`, where the k2 peak
# intensities, their imbalance and their separation trace the chromospheric temperature,
# velocity and velocity gradient.

(mg_ii_core,) = get_lines([wavelength[0], wavelength[-1]], main_only=True)["wavelength"]
line_core = mg_ii_k.crop([SpectralCoord(mg_ii_core), None], [SpectralCoord(mg_ii_core), None])

# `~irispy.utils.fitting.fit_to_maps` gives each component's integrated intensity and
# Doppler velocity as maps, against the Mg II k rest wavelength.
maps = fit_to_maps(mg_ii_model_fit, mg_ii_k, fitter=fitter)
fluxes = maps["integrated_intensity_1"].data, maps["integrated_intensity_2"].data
velocities = maps["velocity_1"].data, maps["velocity_2"].data
# The two Gaussians can swap during the fit, so the blue one is the one at the lower velocity.
swapped = velocities[0] > velocities[1]
blue_flux, red_flux = np.where(swapped, fluxes[1], fluxes[0]), np.where(swapped, fluxes[0], fluxes[1])
# We leave out the fits that failed or stopped at a bound.
valid = (maps["quality"].data == FitQualityFlag.OK) & (blue_flux > 0) & (red_flux > 0)
total_flux = np.where(valid, blue_flux + red_flux, np.nan)
flux_asymmetry = np.where(valid, (blue_flux - red_flux) / total_flux, np.nan)
component_separation = np.where(valid, np.abs(velocities[1] - velocities[0]), np.nan)
flux_unit = maps["integrated_intensity_1"].unit

# The combined maps are plain arrays, so we wrap them in `~irispy.spectrograph.SpectrogramCube`
# objects with the spatial WCS of the maps; they then plot with the same orientation and coordinates.
map_wcs = maps["velocity_1"].wcs
fig, ax_dict = plt.subplot_mosaic(
    [["fov", "total_flux"], ["asymmetry", "separation"]],
    subplot_kw={"projection": line_core.wcs},
    figsize=(12, 10),
)

line_core_max = np.nanpercentile(line_core.data, 99.99)
line_core.plot(axes=ax_dict["fov"], plot_axes=["x", "y"], vmin=0, vmax=line_core_max)
ax_dict["fov"].set_title("Mg II k core")
fig.colorbar(ax_dict["fov"].images[0], ax=ax_dict["fov"], label="Intensity [DN]", shrink=0.8)

for key, data, title, label, symmetric in [
    ("total_flux", total_flux, "Total Gaussian Flux", f"Total flux [{flux_unit.to_string()}]", False),
    # Reversed colours, so that pixels where the blue component is stronger are blue.
    ("asymmetry", flux_asymmetry, "Blue-red Flux Asymmetry", "(blue - red) / total flux", True),
    ("separation", component_separation, "Gaussian Peak Separation", "Peak separation [km / s]", False),
]:
    limit = np.nanpercentile(np.abs(data), 99.99)
    SpectrogramCube(data, map_wcs).plot(
        axes=ax_dict[key],
        plot_axes=["x", "y"],
        cmap="coolwarm_r" if symmetric else None,
        vmin=-limit if symmetric else 0,
        vmax=limit,
    )
    fig.colorbar(ax_dict[key].images[0], ax=ax_dict[key], label=label, extend="both" if symmetric else "neither")
    ax_dict[key].set_title(title)

for ax in ax_dict.values():
    # The first world axis is latitude, along the slit (y), and the second is longitude, along the raster (x).
    for coord, side in ((ax.coords[0], "l"), (ax.coords[1], "b")):
        coord.set_ticklabel(exclude_overlapping=True, fontsize=8)
        coord.set_ticks_position(side)
        coord.set_ticklabel_position(side)
        coord.set_axislabel_position(side)
fig.tight_layout()

plt.show()

# sphinx_gallery_thumbnail_number = 2
