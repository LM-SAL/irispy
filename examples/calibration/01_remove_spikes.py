"""
=====================================
Remove Cosmic Ray Hits from IRIS data
=====================================

This example illustrates how to remove cosmic ray hits from IRIS spectrograph data.
This can also be done for slit-jaw images, but we will focus on spectra as that is a more
common use case.

We will use both ``rsliding`` and ``astroscrappy`` backends, which have to be installed separately using ``pip`` or ``conda``.

To understand both backends, their parameters and how they work, we suggest you read the original
documentation for each package.

* `rsliding documentation <https://git.ias.u-psud.fr/avoyeux/rsliding>`__
* `astroscrappy documentation <https://astroscrappy.readthedocs.io/en/latest/>`__
"""

import matplotlib.pyplot as plt
import pooch

from astropy.visualization import quantity_support

from irispy.io import read_files

quantity_support()

###############################################################################
# `We start with getting data from the IRIS data archive <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20260209_215233_3602506433_2026-02-09T21%3A52%3A332026-02-09T21%3A52%3A33.xml>`__.
#
# This dataset is during the South Atlantic Anomaly (SAA) passage, which is known to cause a large
# number of cosmic ray hits in the data.
#
# This is what we call, a worse-case scenario for cosmic ray removal, which is good for testing the algorithms but not ideal for science.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.
#
# You will need to update the path to the data in the next section if you do that.

raster_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2026/02/09/20260209_215233_3602506433/iris_l2_20260209_215233_3602506433_raster.tar.gz",
    known_hash="bad4a3617d0fd04679203951d7db595df196f79b41a3f6f1f71ce0e301486434",
)

###############################################################################
# We will now open the data using a helper function which is designed to read
# all files from a single observation.

raster = read_files(raster_filename, spectral_windows="Si IV 1403")
# Open the data and select one slice for the comparison.
raster = raster["Si IV 1403"][10][4]

###############################################################################
# Now we use ``remove_cosmic_rays`` .
#
# We suggest you experiment with the parameters to find the best set for your data.
# We have found that the default parameters work well for most data, but in some cases, like this
# one, you may want to be more aggressive in marking cosmic ray hits.

raster_astroscrappy = raster.remove_cosmic_rays(method="astroscrappy")
# We will use a larger kernel for rsliding, which makes it more aggressive in marking cosmic ray
# hits. This will be a consequence of the spectral and spatial resolution of the data.
raster_rsliding = raster.remove_cosmic_rays(method="rsliding", method_kwargs={"kernel": 5})

###############################################################################
# One reason to always be cautious when removing cosmic rays is that you can easily remove real features in the data if you are too aggressive.
# For example, in this case, we have a strong Si IV 1403 line at around row 237, which is removed as a spike by both algorithms.
# But we will look at a different row instead, which has a cleaner profile for Si IV 1403.

si_iv_idx = 231

fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

axes[0].imshow(raster.data, aspect="auto", vmin=0, vmax=500, origin="lower")
axes[0].set_title("Original")
axes[1].imshow(raster_astroscrappy.data, aspect="auto", vmin=0, vmax=500, origin="lower")
axes[1].set_title("astroscrappy")
axes[2].imshow(raster_rsliding.data, aspect="auto", vmin=0, vmax=500, origin="lower")
axes[2].set_title("rsliding")

for ax in axes:
    ax.axhline(si_iv_idx, color="white", linestyle="--", linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

###############################################################################
# Finally, compare the line profile along the marked row.

si_iv_wave = raster.axis_world_coords("wl")[0].to_value("angstrom")

fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
ax.plot(si_iv_wave, raster.data[si_iv_idx, :], label="Original", linestyle="dotted", color="black")
ax.plot(si_iv_wave, raster_astroscrappy.data[si_iv_idx, :], label="astroscrappy", linestyle="dashed")
ax.plot(si_iv_wave, raster_rsliding.data[si_iv_idx, :], label="rsliding", linestyle="dashed")
ax.set_xlabel("Wavelength (Å)")
ax.set_xlim(1400, 1406)
ax.legend()

###############################################################################
# In practice, the comparison is not about choosing one universally "better"
# algorithm. You will need to run both, change the parameters, and visually
# inspect the results to find the best solution for your data and science case.

plt.show()
