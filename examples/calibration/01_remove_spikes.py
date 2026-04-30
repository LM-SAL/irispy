"""
=====================================
Remove Cosmic Ray Hits from IRIS data
=====================================

This example illustrates how to remove cosmic ray hits from IRIS data.

We will use the ``astroscrappy`` backend and has to be installed separately using ``pip`` or ``conda``.

``irispy`` supports two backends for cosmic ray removal, they are compared in this example: :ref:`sphx_glr_generated_gallery_calibration_04_compare_cosmic_ray_backends.py`.
"""

import matplotlib.pyplot as plt
import pooch

from astropy.visualization import quantity_support

from irispy.io import read_files

quantity_support()

###############################################################################
# `We start with getting data from the IRIS data archive <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20260209_215233_3602506433_2026-02-09T21%3A52%3A332026-02-09T21%3A52%3A33.xml>`__.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.
#
# You will need to update the path to the data in the next section if you do that.

sji_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2026/02/06/20260206_210853_3460104433/iris_l2_20260206_210853_3460104433_SJI_2832_t000.fits.gz",
    known_hash="d5088c6a0753ea9ce7b525865ba2edf13a637097ee995985b511833897c88ca6",
)
raster_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2026/02/09/20260209_215233_3602506433/iris_l2_20260209_215233_3602506433_raster.tar.gz",
    known_hash="bad4a3617d0fd04679203951d7db595df196f79b41a3f6f1f71ce0e301486434",
)

###############################################################################
# We will now open the data using a helper function which is designed to read
# all files from a single observation.

sji_2832 = read_files(sji_filename)
raster = read_files(raster_filename)
raster_2796 = raster["Mg II k 2796"][10][4]
# Used to reduce memory usage for the online documentation build.
del raster

###############################################################################
# Now we use ``remove_cosmic_rays`` with the ``astroscrappy`` backend.
#
# This algorithm can perform well with both high and low noise levels in the original data.
# This particular image has lots of high intensity cosmic ray hits which
# cannot be effectively removed by using the default set of parameters.
# So we reduce ``sigma`` (mapped to astroscrappy's ``sigclip``) from 4.5 to 2 to mark more hits.
# We also reduce ``objlim``, the contrast between the Laplacian image and the fine structured image
# to clean the high intensity bright cosmic ray hits.
# We also modify the ``readnoise`` parameter to obtain better results.
#
# For more details on the parameters, see the `astroscrappy documentation <https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html#astroscrappy.detect_cosmics>`__.

sji_cleaned = sji_2832.remove_cosmic_rays(
    method="astroscrappy",
    sigma=2,
    method_kwargs={"objlim": 2, "readnoise": 4},
)

###############################################################################
# ``remove_cosmic_rays`` returns a cleaned cube with the same metadata and coordinates.
# We now convert a noisy frame to a map for plotting.

sji_map = sji_2832.to_maps(5)
clean_sji_map = sji_cleaned.to_maps(5)

fig = plt.figure(figsize=(12, 5), constrained_layout=True)

ax = fig.add_subplot(121, projection=sji_map)
sji_map.plot(axes=ax, vmin=0, vmax=500)
ax.set_title("Original")

ax1 = fig.add_subplot(122, projection=clean_sji_map)
clean_sji_map.plot(axes=ax1, vmin=0, vmax=500)
ax1.set_title("Cleaned")

ax1.coords[1].set_ticks_visible(False)
ax1.coords[1].set_ticklabel_visible(False)

###############################################################################
# For comparison, we will now try to remove the cosmic ray hits from the
# spectrograph data.

raster_2796_cleaned = raster_2796.remove_cosmic_rays(method="astroscrappy")

fig = plt.figure(figsize=(12, 5), constrained_layout=True)
axes = [
    fig.add_subplot(1, 2, 1, projection=raster_2796.wcs),
    fig.add_subplot(1, 2, 2, projection=raster_2796_cleaned.wcs),
]

raster_2796.plot(axes=axes[0], vmin=0, vmax=500)
axes[0].set_title("Original Mg II k 2796")

raster_2796_cleaned.plot(axes=axes[1], vmin=0, vmax=500)
axes[1].set_title("Cleaned Mg II k 2796")

for ax in axes:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

###############################################################################
# For any cosmic ray removal, it is important to check the results, especially
# if you are interested in the spectral data. These methods are not perfect and
# can sometimes mark real features as cosmic ray hits, especially if the
# parameters are not tuned for the specific data. Always check the results
# to ensure that important features are not being removed.

plt.show()
