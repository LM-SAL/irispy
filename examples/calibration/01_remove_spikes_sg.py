"""
====================================
Remove Cosmic Rays from IRIS SG data
====================================

This example illustrates how to remove cosmic ray hits from IRIS spectrograph data.

We will use the ``rsliding`` backend, which has to be installed separately with ``pip`` or ``conda``
and is the better choice for spectral data. See the
`rsliding documentation <https://git.ias.u-psud.fr/avoyeux/rsliding>`__ for how it works and its parameters.
"""

import matplotlib.pyplot as plt
import pooch

from astropy.visualization import quantity_support

from irispy.io import read_files

quantity_support()

###############################################################################
# `We start with getting data from the IRIS data archive <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20260209_215233_3602506433_2026-02-09T21%3A52%3A332026-02-09T21%3A52%3A33.xml>`__.
# The full observation is available as a `Level 2 raster tarball <https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2026/02/09/20260209_215233_3602506433/iris_l2_20260209_215233_3602506433_raster.tar.gz>`__.
# To keep the download small, we use a cutout of it that only has the eleventh raster scan of the Si IV 1403 window.
#
# This dataset was taken during a South Atlantic Anomaly (SAA) passage, so it has many
# cosmic ray hits: a worst case, good for testing the algorithm but not ideal for science.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.
#
# You will need to update the path to the data in the next section if you do that.

raster_filename = pooch.retrieve(
    "https://github.com/LM-SAL/irispy-data/releases/download/v1/iris_l2_20260209_215233_3602506433_cutout_raster.tar.gz",
    known_hash="cfed302a860202b1a8fdd90d3f1faec23bfca60a231bbbb9b851ea0c57d14fd8",
)

###############################################################################
# We will now open the data using a helper function which is designed to read
# all files from a single observation.

raster = read_files(raster_filename, spectral_windows="Si IV 1403")
# Open the data and select one slice for the comparison.
raster = raster["Si IV 1403"][0][4]

###############################################################################
# Now we use ``remove_cosmic_rays`` with the ``rsliding`` backend, the default, which the
# SPICE team uses for their data. Its defaults are not tuned for IRIS and no single setting
# suits every dataset, so read its documentation and experiment.
#
# The main parameter is the ``kernel`` size, which sets how aggressive the despiking is and
# depends on the spectral sampling. For ~0.0254 Å pixels, 3 (the default) misses some spikes
# but tolerates real jumps in the continuum, while 5 catches most spikes but sometimes replaces
# real features; this dataset is binned to ~0.051 Å per pixel. ``threads`` sets how many CPU
# cores are used.

# These settings smooth more around each spike (thanks to Juraj).
method_kwargs = {"kernel": 5, "center_choice": "median", "borders": "reflect"}
raster_rsliding = raster.remove_cosmic_rays(method="rsliding", sigma=3, method_kwargs=method_kwargs)

###############################################################################
# One reason to always be cautious when removing cosmic rays is that you can
# easily remove real features in the data if you are too aggressive.
# For example, a strong and narrow line can look like a spike to the algorithm.
# Here we compare a row with a clean Si IV 1403 profile.

si_iv_idx = 66

fig, axes = plt.subplots(1, 2, figsize=(10, 4), subplot_kw={"projection": raster.wcs})

raster.plot(axes=axes[0], aspect="auto", vmin=0, vmax=500)
axes[0].set_title("Original")
raster_rsliding.plot(axes=axes[1], aspect="auto", vmin=0, vmax=500)
axes[1].set_title("rsliding")

for ax in axes:
    ax.axhline(si_iv_idx, color="white", linestyle="--", linewidth=1)
    # Longitude barely changes along the slit, so we hide its axis.
    longitude = ax.coords["custom:pos.helioprojective.lon"]
    longitude.set_ticks_visible(False)
    longitude.set_ticklabel_visible(False)
    longitude.set_axislabel("")

###############################################################################
# Finally, compare the line profile along the marked row.

si_iv_wave = raster.axis_world_coords("wl")[0].to_value("angstrom")

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(si_iv_wave, raster.data[si_iv_idx, :], label="Original", linestyle="dotted", color="black")
ax.plot(si_iv_wave, raster_rsliding.data[si_iv_idx, :], label="rsliding", linestyle="dashed")
ax.set_ylabel("Intensity (DN)")
ax.set_xlabel("Wavelength (Å)")
ax.set_xlim(1400, 1406)
ax.legend()

plt.show()
