"""
==============================
Fit Spectral Models to Spectra
==============================

In this example, we are going to fit Si IV 1403 from IRIS with a single Gaussian, starting
from `~irispy.utils.fitting.si_iv_1403_model`. Then we will use the fitted values to make
maps of the line's flux, Doppler shift and width.

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
from astropy import constants
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.modeling.fitting import TRFLSQFitter, parallel_fit_dask
from astropy.wcs.utils import wcs_to_celestial_frame

from sunpy.coordinates.frames import Helioprojective

from irispy.io import read_files
from irispy.spectrograph import SpectrogramCube
from irispy.utils.fitting import si_iv_1403_model

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
# neighbouring O IV lines at 1401.157 and 1404.806 Å (De Pontieu et al. 2014; Polito et al. 2016).

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
iris_model_fit = parallel_fit_dask(
    model=model,
    fitter=TRFLSQFitter(),
    data=np.where(good, si_iv_1403.data, 0),
    data_unit=si_iv_1403.unit,
    weights=np.where(good, 1 / si_iv_1403.uncertainty.array, 0),
    world=(wavelength,),
    fitting_axes=2,
    scheduler="single-threaded",
)

###############################################################################
# The fitted parameters are 2D arrays with the shape of the spatial axes. We convert
# them into physical quantities and wrap them in `~irispy.spectrograph.SpectrogramCube`
# objects with the WCS of the line-core image, so that they plot with the same
# orientation and coordinates.

fig, ax_dict = plt.subplot_mosaic(
    [["fov", "net_flux"], ["velocity", "sigma"]],
    subplot_kw={"projection": si_iv_spec_crop.wcs},
    figsize=(12, 10),
)

si_iv_spec_crop.plot(axes=ax_dict["fov"], plot_axes=["x", "y"], vmin=0, vmax=200)
ax_dict["fov"].set_title(f"Si IV {si_iv_core.to_value(u.AA)} Å")
fig.colorbar(ax_dict["fov"].images[0], ax=ax_dict["fov"], label="Intensity [DN]", shrink=0.8)

gaussian_width = iris_model_fit.stddev_1.quantity
net_flux = (
    np.sqrt(2 * np.pi) * iris_model_fit.amplitude_1.quantity * gaussian_width / np.mean(np.diff(wavelength))
).to(si_iv_1403.unit)
amp_max = np.nanpercentile(np.abs(net_flux.value), 99)
SpectrogramCube(net_flux, si_iv_spec_crop.wcs).plot(
    axes=ax_dict["net_flux"], plot_axes=["x", "y"], vmin=0, vmax=amp_max
)
cbar = fig.colorbar(ax_dict["net_flux"].images[0], ax=ax_dict["net_flux"])
cbar.set_label(label=f"Intensity [{net_flux.unit.to_string()}]", fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax_dict["net_flux"].set_title("Gaussian Net Flux")

core_shift = ((iris_model_fit.mean_1.quantity - si_iv_core) / si_iv_core * constants.c).to(u.km / u.s)
shift_max = np.nanpercentile(np.abs(core_shift.value), 95)
SpectrogramCube(core_shift, si_iv_spec_crop.wcs).plot(
    axes=ax_dict["velocity"], plot_axes=["x", "y"], cmap="coolwarm", vmin=-shift_max, vmax=shift_max
)
cbar = fig.colorbar(ax_dict["velocity"].images[0], ax=ax_dict["velocity"], extend="both")
cbar.set_label(label=f"Doppler shift [{core_shift.unit.to_string()}]", fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax_dict["velocity"].set_title("Velocity from Gaussian shift")

sigma = (gaussian_width / si_iv_core * constants.c).to(u.km / u.s)
line_max = np.nanpercentile(np.abs(sigma.value), 95)
SpectrogramCube(sigma, si_iv_spec_crop.wcs).plot(axes=ax_dict["sigma"], plot_axes=["x", "y"], vmax=line_max)
cbar = fig.colorbar(ax_dict["sigma"].images[0], ax=ax_dict["sigma"])
cbar.set_label(label=f"Line Width [{sigma.unit.to_string()}]", fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax_dict["sigma"].set_title("Gaussian Sigma")

for ax in ax_dict.values():
    # The first world axis is latitude, along the slit (y), and the second is longitude, along the raster (x).
    for coord, side in ((ax.coords[0], "l"), (ax.coords[1], "b")):
        coord.set_ticklabel(exclude_overlapping=True, fontsize=8)
        coord.set_ticks_position(side)
        coord.set_ticklabel_position(side)
        coord.set_axislabel_position(side)
fig.tight_layout()

plt.show()
