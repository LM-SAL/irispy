"""
==============================
Fit Spectral Models to Spectra
==============================

In this example, we are going to fit Si IV 1403 from IRIS with a single Gaussian.
Then we will use the fitted values to calculate the Gaussian moments.

For a model-independent alternative, the spectral moments, see
:ref:`sphx_glr_generated_gallery_analysis_04_spectral_moments.py`.

If you want to see a similar example but with a double Gaussian fit to the Mg II k line,
see :ref:`sphx_glr_generated_gallery_analysis_07_mg_ii_two_gaussian_fitting.py`.
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pooch

import astropy.units as u
from astropy import constants
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.modeling import models as m
from astropy.modeling.fitting import LMLSQFitter, TRFLSQFitter, parallel_fit_dask
from astropy.wcs.utils import wcs_to_celestial_frame

from sunpy.coordinates.frames import Helioprojective

from irispy.io import read_files
from irispy.spectrograph import SpectrogramCube

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
# We read only the Si IV 1403 window and select the one complete scan.

raster = read_files(raster_filename, spectral_windows="Si IV 1403")
si_iv_1403 = raster["Si IV 1403"][0]

###############################################################################
# Before we get to fitting, we will shrink the data cube to make it easier to work with.
# This is done primarily to speed up the fitting process on the online documentation build.

iris_observer = wcs_to_celestial_frame(si_iv_1403.wcs.celestial).observer
iris_frame = Helioprojective(observer=iris_observer)
top_left = [None, SkyCoord(-360 * u.arcsec, 310 * u.arcsec, frame=iris_frame)]
bottom_right = [None, SkyCoord(-290 * u.arcsec, 260 * u.arcsec, frame=iris_frame)]
si_iv_1403 = si_iv_1403.crop(top_left, bottom_right)
# We also average 2x2 spatial pixels, after trimming both spatial axes to an even length.
# This improves the signal-to-noise of the faint Si IV line and means 4x fewer fits.
ny, nx = (n // 2 * 2 for n in si_iv_1403.data.shape[:2])
si_iv_1403 = si_iv_1403[:ny, :nx].rebin((2, 2, 1))

###############################################################################
# Let us just get the full field of view at the line core.

si_iv_core = 140.277 * u.nm
lower_corner = [SpectralCoord(si_iv_core), None]
upper_corner = [SpectralCoord(si_iv_core), None]
si_iv_spec_crop = si_iv_1403.crop(lower_corner, upper_corner)

###############################################################################
# We will want the spectrum averaged over all spatial pixels.

spatial_mean = si_iv_1403.rebin((*si_iv_1403.data.shape[:-1], 1))[0, 0, :]
wavelength_coords = spatial_mean.axis_world_coords("em.wl")[0].to(u.nm)

###############################################################################
# We fit the data in DN, without radiometric calibration. The initial model is a
# constant plus a Gaussian. You can pick any constant such that spurious values
# in the core do not skew the first guess. Here, we use amplitude that is the 10th
# percentile of the non-core window.

si_iv_core_window = np.abs(wavelength_coords - si_iv_core) < 0.15 * u.nm
initial_model = m.Const1D(
    amplitude=np.nanpercentile(spatial_mean.data[~si_iv_core_window], 10) * si_iv_1403.unit
) + m.Gaussian1D(
    amplitude=np.nanmax(spatial_mean.data[si_iv_core_window]) * si_iv_1403.unit, mean=si_iv_core, stddev=0.005 * u.nm
)

###############################################################################
# To improve the initial guess, we fit the initial model to the spatially averaged
# spectrum, using the wavelengths from `ndcube.NDCube.axis_world_coords`.

fitter = TRFLSQFitter()
average_fit = fitter(
    initial_model,
    wavelength_coords,
    spatial_mean.data * spatial_mean.unit,
)

###############################################################################
# Now we compare the initial model with the model fitted to the average spectrum.

fig = plt.figure()
ax = spatial_mean.plot(label="Spatial average")
ax.plot(initial_model(wavelength_coords), label="Initial model")
ax.plot(average_fit(wavelength_coords), linestyle="--", label="Spatial average fit")
plt.legend()

###############################################################################
# `~astropy.modeling.fitting.parallel_fit_dask` fits the model to every spectrum
# along the fitting axis, here the wavelength axis, and returns a model whose
# parameters are arrays with the shape of the other axes. Its documentation
# describes the arguments; the data can be an array or an `~astropy.nddata.NDData`,
# whose WCS, mask and unit are then used.

# Basic data sanitization: set negative and non-finite values to zero.
filtered_data = np.where(si_iv_1403.data < 0, 0, si_iv_1403.data)
filtered_data = np.where(np.isfinite(filtered_data), filtered_data, 0)

###############################################################################
# Fits that fail, usually because they do not converge, do not raise:
# `~astropy.modeling.fitting.parallel_fit_dask` sets the parameters of that pixel
# to NaN. To see why, set the ``diagnostics`` and ``diagnostics_path`` keyword arguments.

diag_path = Path("./diag")
shutil.rmtree(diag_path, ignore_errors=True)

# Now we fit the cube.
iris_model_fit = parallel_fit_dask(
    data=filtered_data,
    data_unit=si_iv_1403.unit,
    fitting_axes=2,
    # We are fitting along the wavelength axis, so we need to provide the world coordinates
    # along this axis. The input has to be a tuple of length equal to the number of fitting axes.
    world=(wavelength_coords,),
    model=average_fit,
    # You can replace this with TRFLSQFitter; LMLSQFitter is faster in a single thread,
    # which is why we use it here.
    fitter=LMLSQFitter(),
    scheduler="single-threaded",
    # See above for the error handling discussion
    diagnostics="error",
    diagnostics_path=diag_path,
)

###############################################################################
# This example fits in a single thread. To use several cores, pass a dask client
# as the scheduler instead:
#
# .. code-block:: python
#
#     from dask.distributed import Client
#
#     scheduler=Client(),
#
# Now let us check for errors during the fit, which are written to the "diag" folder.

errors = [p.read_text() for p in diag_path.rglob("error.log")]
print(f"{len(errors)} errors occurred")
if errors:
    print("First error is:")
    print(errors[0])

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
ax_dict["fov"].set_title("Si IV 1402.77 Å")
fig.colorbar(ax_dict["fov"].images[0], ax=ax_dict["fov"], label="Intensity [DN]", shrink=0.8)

# The fitter does not keep the Gaussian width positive, so a few fits return a negative one.
# Only its size matters, so we use its absolute value.
gaussian_width = np.abs(iris_model_fit.stddev_1.quantity)
net_flux = (
    np.sqrt(2 * np.pi)
    * (iris_model_fit.amplitude_1)
    * gaussian_width
    / np.mean(si_iv_1403.axis_world_coords("wl")[0][1:] - si_iv_1403.axis_world_coords("wl")[0][:-1]).to(u.nm)
)
amp_max = np.nanpercentile(np.abs(net_flux.value), 99)
SpectrogramCube(net_flux, si_iv_spec_crop.wcs).plot(
    axes=ax_dict["net_flux"], plot_axes=["x", "y"], vmin=0, vmax=amp_max
)
cbar = fig.colorbar(ax_dict["net_flux"].images[0], ax=ax_dict["net_flux"])
cbar.set_label(label=f"Intensity [{net_flux.unit.to_string()}]", fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax_dict["net_flux"].set_title("Gaussian Net Flux")

core_shift = ((iris_model_fit.mean_1.quantity.to(u.nm)) - si_iv_core) / si_iv_core * (constants.c.to(u.km / u.s))
shift_max = np.nanpercentile(np.abs(core_shift.value), 95)
SpectrogramCube(core_shift, si_iv_spec_crop.wcs).plot(
    axes=ax_dict["velocity"], plot_axes=["x", "y"], cmap="coolwarm", vmin=-shift_max, vmax=shift_max
)
cbar = fig.colorbar(ax_dict["velocity"].images[0], ax=ax_dict["velocity"], extend="both")
cbar.set_label(label=f"Doppler shift [{core_shift.unit.to_string()}]", fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax_dict["velocity"].set_title("Velocity from Gaussian shift")

sigma = gaussian_width.to(u.nm) / si_iv_core * (constants.c.to(u.km / u.s))
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

# sphinx_gallery_thumbnail_number = 2
