"""
=============================
Apply Radiometric Calibration
=============================

In this example we will show how to perform radiometric calibration on IRIS data.

IRIS Level 2 data are provided in units of Data Number (DN). To convert these to a flux
in physical units (e.g., :math:`erg s^{-1} sr^{-1} cm^{-2} Å^{-1}`) one must perform a
radiometric calibration.

The output is in the units of SolarSoft's
`iris_calib_spectrum.pro <https://hesperia.gsfc.nasa.gov/ssw/iris/idl/lmsal/iris2/iris_calib_spectrum.pro>`__,
radiance per Å. The NRL routine `IRIS_CALIB <https://hesperia.gsfc.nasa.gov/ssw/iris/idl/nrl/iris_calib.pro>`__
takes the DN summed over a line and gives its intensity in :math:`erg s^{-1} sr^{-1} cm^{-2}`: to compare,
sum the calibrated spectrum over the line and multiply by ``abs(cube.spectral_dispersion)``. See
`ITN26 <https://iris.lmsal.com/itn26/calibration.html>`__ for the calibration process.
"""

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

from irispy.data import sample
from irispy.io import read_files
from irispy.utils.spectrograph import radiometric_calibration

###############################################################################
# We start with the raster sample file, which ``irispy.data.sample`` downloads on first use.
# To use your own data, pass the path of a Level 2 raster tarball from the
# `IRIS data archive <https://www.lmsal.com/hek/hcr>`__ to ``read_files`` instead.

raster_filename = sample.RASTER_TAR

###############################################################################
# We will now open the data using a helper function which is designed to read
# all files from a single observation.

raster = read_files(raster_filename, spectral_windows=["Mg II k 2796", "Si IV 1394"])

###############################################################################
# We will focus on the Mg II k 2796 (NUV) and Si IV 1394 (FUV) lines which we
# can select using a key. Then we will just plot a spectral line selected at
# random in space.

# To keep the example light enough for the online documentation build, we take
# the first raster step of the first raster only.
mg_ii_k_2796 = raster["Mg II k 2796"][0, 0]
si_iv_1394 = raster["Si IV 1394"][0, 0]
del raster

###############################################################################
# To convert the spectral units from DN to flux one must do the following calculation:
#
# .. math::
#
#    \mathrm{Flux}(\mathrm{erg}\: \mathrm{s}^{-1}\: \mathrm{cm}^{-2} \text{Å}^{-1}\: \mathrm{sr}^{-1}) = \mathrm{Flux}(\mathrm{DN}) \frac{E_\lambda \cdot \mathrm{DN2PHOT\_SG}}{A_\mathrm{eff} \cdot \mathrm{Pix}_{xy} \cdot \mathrm{Pix}_{\lambda} \cdot t_\mathrm{exp} \cdot W_\mathrm{slit}},
#
# where :math:`E_\lambda \equiv h \cdot c / \lambda` is the photon energy (in erg),
# :math:`DN2PHOT\_SG` is the number of photons per DN,
# :math:`A_\mathrm{eff}` is the effective area (in :math:`cm^{2}`),
# :math:`Pix_{xy}` is the size of the spatial pixels in radians (e.g., multiply the spatial binning factor by :math:`\pi/(180\cdot3600\cdot6)`),
# :math:`Pix_{\lambda}` is the size of the spectral pixels in :math:`Å`,
# :math:`t_\mathrm{exp}` is the exposure time in seconds and
# :math:`W_\mathrm{slit}` is the slit width in radians (:math:`W_\mathrm{slit} \equiv \pi/(180\cdot3600\cdot3)`).
#
# `irispy.utils.spectrograph.radiometric_calibration` does this calculation, with the units handled.

calibrated_mg_ii_k_2796 = radiometric_calibration(mg_ii_k_2796)
calibrated_si_iv_1394 = radiometric_calibration(si_iv_1394)

###############################################################################
# We will now plot the spectra before and after at a single spatial pixel,
# applying the cube's mask to remove bad pixels.


def plot_before_after(cube, calibrated_cube, title, spectral_slice=slice(None)):
    fig, ax = plt.subplots()
    color = "tab:red"
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Counts (DN)", color=color)
    ax.plot(
        cube.spectral_axis[spectral_slice].to(u.angstrom),
        np.ma.masked_where(cube.mask, cube.data)[100, spectral_slice],
        color=color,
        linestyle="dashed",
    )
    ax.tick_params(axis="y", labelcolor=color)
    ax2 = ax.twinx()
    color = "tab:blue"
    ax2.plot(
        calibrated_cube.spectral_axis[spectral_slice].to(u.angstrom),
        np.ma.masked_where(calibrated_cube.mask, calibrated_cube.data)[100, spectral_slice],
        color=color,
    )
    ax2.set_ylabel("Calibrated Intensity (erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ Å$^{-1}$)", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax.set_title(title)
    fig.tight_layout()
    return fig


plot_before_after(mg_ii_k_2796, calibrated_mg_ii_k_2796, "Mg II k 2796 Spectrum", spectral_slice=slice(10, -20))

###############################################################################
# Now the same spatial pixel for the Si IV 1394 (FUV) window.
#
# This window extends blueward of the nominal FUV range (~1389-1408 Å) over which the
# response file defines the effective area, so the calibrated (blue) curve stops at
# ~1389 Å while the counts (red) continue to the edge of the window.

plot_before_after(si_iv_1394, calibrated_si_iv_1394, "Si IV 1394 Spectrum")

###############################################################################
# `~irispy.utils.spectrograph.radiometric_calibration` returns radiance per unit wavelength.
# `ndcube.NDCube.to` converts the data and uncertainty to other units. Per unit frequency
# or photon units depend on wavelength, so they also need `astropy.units.spectral_density`
# with the cube's wavelengths.

wavelength = calibrated_mg_ii_k_2796.axis_world_coords("em.wl")[0]
per_hz = calibrated_mg_ii_k_2796.to(u.erg / u.cm**2 / u.s / u.sr / u.Hz, equivalencies=u.spectral_density(wavelength))
print(per_hz.unit)

plt.show()
