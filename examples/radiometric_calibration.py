"""
=======================
Radiometric Calibration
=======================

In this example we will show how to perform radiometric calibration on IRIS data.

IRIS level 2 data are provided in units of Data Number (DN).
To convert these to a flux in physical units (e.g., :math:`erg s^{-1} sr^{-1} cm^{-2} Å^{-1}`)
one must perform a radiometric calibration.

Please refer to `ITN26 for more information on the calibration process <https://iris.lmsal.com/itn26/calibration.html>`__.
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
# Using the url: http://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2018/01/02/20180102_153155_3610108077/iris_l2_20180102_153155_3610108077_raster.tar.gz
# we are after the raster sequence (~300 MB).
#
# The full observation is at https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20180102_153155_3610108077_2018-01-02T15%3A31%3A552018-01-02T15%3A31%3A55.xml
#
raster_filename = pooch.retrieve(
    "http://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2018/01/02/20180102_153155_3610108077/iris_l2_20180102_153155_3610108077_raster.tar.gz",
    known_hash="8949562149cfa5fba067b5b102e8434b14cea3c3416dd79c06b7f6e211c61a39",
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
# We will just focus on the Si IV 1403 line which we can select using a key.
# Then we will just plot a spectral line selected at random in space.

# There is only one complete scan, so we index that away.
si_iv_1403 = raster["Si IV 1403"][0]

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

calibrated_si_iv_1403 = radiometric_calibration(si_iv_1403)

###############################################################################
# We will now plot both the before and after spectrums to see the difference.
# We will plot the spectrum at a single spatial pixel.

fig, ax = plt.subplots()
color = "tab:red"
ax.set_xlabel("Wavelength (Å)")
ax.set_ylabel("Counts (DN)", color=color)
ax.plot(
    si_iv_1403.spectral_axis[10:-20].to(u.angstrom), si_iv_1403.data[100, 100, 10:-20], color=color, linestyle="dashed"
)
ax.tick_params(axis="y", labelcolor=color)

ax2 = ax.twinx()

color = "tab:blue"
ax2.plot(
    calibrated_si_iv_1403.spectral_axis[10:-20].to(u.angstrom),
    calibrated_si_iv_1403.data[100, 100, 10:-20],
    color=color,
)
ax2.set_ylabel("Calibrated Intensity (erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ Å$^{-1}$)", color=color)
ax2.tick_params(axis="y", labelcolor=color)

ax.set_title("Si IV 1403 spectrum before and after radiometric calibration")
fig.tight_layout()

plt.show()
