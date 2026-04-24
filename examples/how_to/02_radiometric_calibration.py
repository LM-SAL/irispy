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
import pooch

import astropy.units as u

from irispy.io import read_files
from irispy.utils.spectrograph import radiometric_calibration

###############################################################################
# We will start by getting some data from the IRIS archive.
#
# In this case, we will use ``pooch`` so to keep this example self-contained
# but using your browser will also work.
#
# Using the url: https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2026/03/08/20260308_051050_3893012099/iris_l2_20260308_051050_3893012099_raster.tar.gz
# we are after the raster observation (~300 MB).
#
# The full observation is at https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20260308_051050_3893012099_2026-03-08T05%3A10%3A502026-03-08T05%3A10%3A50.xml
#
raster_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2026/03/08/20260308_051050_3893012099/iris_l2_20260308_051050_3893012099_raster.tar.gz",
    known_hash="b76277ae89e79f50e7ddc603a86b9bbf70e23b2e64fc5343dbe99af19a05f854",
)

###############################################################################
# Now to open the files using ``irispy``.

# Note that when ``memmap=True``, the data values are read from the FITS file
# directly without the scaling to Float32 (via "b_zero" and "b_scale"),
# the data values are no longer in DN, but in scaled integer units that start at -2$^{16}$/2.
#
# We will use ``memmap=False`` because we want to fit the actual the data values.

raster = read_files(raster_filename, memmap=False)

###############################################################################
# We will just focus on the Mg II k 2796 line which we can select using a key.
# Then we will just plot a spectral line selected at random in space.

# This observation contains one complete raster, so the window value is already a
# single `SpectrogramCube`.
mg_ii_k_2796 = raster["Mg II k 2796"]

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

###############################################################################
# We will now plot both the before and after spectrums to see the difference.
# We will plot the spectrum at a single spatial pixel.

fig, ax = plt.subplots()
color = "tab:red"
ax.set_xlabel("Wavelength (Å)")
ax.set_ylabel("Counts (DN)", color=color)
scan_index = mg_ii_k_2796.shape[0] // 2
slit_index = mg_ii_k_2796.shape[1] // 2
ax.plot(
    mg_ii_k_2796.spectral_axis[10:-20].to(u.angstrom),
    mg_ii_k_2796.data[scan_index, slit_index, 10:-20],
    color=color,
    linestyle="dashed",
)
ax.tick_params(axis="y", labelcolor=color)

ax2 = ax.twinx()

color = "tab:blue"
ax2.plot(
    calibrated_mg_ii_k_2796.spectral_axis[10:-20].to(u.angstrom),
    calibrated_mg_ii_k_2796.data[scan_index, slit_index, 10:-20],
    color=color,
)
ax2.set_ylabel("Calibrated Intensity (erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ Å$^{-1}$)", color=color)
ax2.tick_params(axis="y", labelcolor=color)

ax.set_title("Mg II k 2796 Spectrum")
fig.tight_layout()

plt.show()
