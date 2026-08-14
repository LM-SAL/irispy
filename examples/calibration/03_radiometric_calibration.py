"""
=============================
Apply Radiometric Calibration
=============================

In this example we will show how to perform radiometric calibration on IRIS data.

IRIS level 2 data are provided in units of Data Number (DN). To convert these to a flux
in physical units (e.g., :math:`erg s^{-1} sr^{-1} cm^{-2} Å^{-1}`) one must perform a
radiometric calibration.

The calibration output has been confirmed to provide the same results as those provided
by the SolarSoft IDL routine `IRIS_CALIB <https://hesperia.gsfc.nasa.gov/ssw/iris/idl/nrl/iris_calib.pro>`__.

The major difference being that the output here is accounting for the wavelength, which is why the units
here are :math:`erg s^{-1} sr^{-1} cm^{-2} Å^{-1}` and not :math:`erg s^{-1} sr^{-1} cm^{-2}`.
Notice the extra :math:`Å^{-1}` in the units.

Please refer to
`ITN26 for more information on the calibration process <https://iris.lmsal.com/itn26/calibration.html>`__.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch

import astropy.units as u

from irispy.io import read_files
from irispy.utils.spectrograph import radiometric_calibration

###############################################################################
# `We start with getting data from the IRIS data archive <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20260308_051050_3893012099_2026-03-08T05%3A10%3A502026-03-08T05%3A10%3A50.xml>`__.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.
#
# You will need to update the path to the data in the next section if you do that.

raster_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2026/03/08/20260308_051050_3893012099/iris_l2_20260308_051050_3893012099_raster.tar.gz",
    known_hash="b76277ae89e79f50e7ddc603a86b9bbf70e23b2e64fc5343dbe99af19a05f854",
)

###############################################################################
# We will now open the data using a helper function which is designed to read
# all files from a single observation.

raster = read_files(raster_filename, spectral_windows=["Mg II k 2796", "Si IV 1394"])

###############################################################################
# We will focus on the Mg II k 2796 (NUV) and Si IV 1394 (FUV) lines which we
# can select using a key. Then we will just plot a spectral line selected at
# random in space.

# There is only one complete scan, so we index that away.
# We also only take the first scan of the sequence to reduce memory usage for
# the online documentation build.
mg_ii_k_2796 = raster["Mg II k 2796"][0][0]
si_iv_1394 = raster["Si IV 1394"][0][0]

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
# :math:`A_\mathrm{eff}` is the effective area (in :math:`cm^{-2}`),
# :math:`Pix_{xy}` is the size of the spatial pixels in radians (e.g., multiply the spatial binning factor by :math:`\pi/(180\cdot3600\cdot6)`),
# :math:`Pix_{\lambda}` is the size of the spectral pixels in :math:`Å`,
# :math:`t_\mathrm{exp}` is the exposure time in seconds and
# :math:`W_\mathrm{slit}` is the slit width in radians (:math:`W_\mathrm{slit} \equiv \pi/(180\cdot3600\cdot3)`).
#
# This is a complex equation and requires careful attention to units.
# Within `irispy`, there is a function called `irispy.utils.spectrograph.radiometric_calibration` that handles this process.

calibrated_mg_ii_k_2796 = radiometric_calibration(mg_ii_k_2796)
calibrated_si_iv_1394 = radiometric_calibration(si_iv_1394)

###############################################################################
# We will now plot both the before and after spectrums at a single spatial
# pixel to see the difference.
#
# We will apply the cube's mask when plotting, which removes bad pixels.


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
# This spectral window extends blueward of the nominal FUV spectral range
# (~1389-1408 Å) within which the response file defines the effective area.
#
# `irispy` returns NaN for those wavelengths,
# which is why the calibrated (blue) curve stops at ~1389 Å while the uncalibrated
# counts (red) continue to the edge of the window.

plot_before_after(si_iv_1394, calibrated_si_iv_1394, "Si IV 1394 Spectrum")

plt.show()
