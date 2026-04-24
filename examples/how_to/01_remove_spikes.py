"""
=====================================
Remove Cosmic Ray Hits from IRIS data
=====================================

This example illustrates how to remove cosmic ray hits from a IRIS SJI FITS file
using the built-in ``remove_cosmic_rays()`` API in ``irispy``.

``astroscrappy`` is a separate Python package and can be installed separately using ``pip`` or ``conda``.

The ``astroscrappy`` backend works on one 2D frame at a time and does not use
temporal context from a time series.
"""

import matplotlib.pyplot as plt
import pooch

from astropy.visualization import quantity_support

from irispy.io import read_files

quantity_support()

###############################################################################
# We start with getting the data.
# This is done by downloading the data from the IRIS archive.
#
# In this case, we will use ``pooch`` as to keep this example self-contained
# but using your browser will also work.
#
# The data is from a `BBSO coordination on the 6th February 2026 <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20260209_215233_3602506433_2026-02-09T21%3A52%3A332026-02-09T21%3A52%3A33.xml>`__, and is a SJI image in 2832 Å.

sji_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2026/02/06/20260206_210853_3460104433/iris_l2_20260206_210853_3460104433_SJI_2832_t000.fits.gz",
    known_hash="d5088c6a0753ea9ce7b525865ba2edf13a637097ee995985b511833897c88ca6",
)

###############################################################################
# We will now open the file we just downloaded and select one frame.
#
# ``astroscrappy`` works on one 2D frame at a time. Selecting a single frame
# keeps this example lightweight enough for documentation builds while still
# demonstrating the same API.

sji_2832 = read_files(sji_filename, memmap=False)
sji_frame = sji_2832[5]
del sji_2832

###############################################################################
# Now we use ``SJICube.remove_cosmic_rays`` with the ``astroscrappy`` backend.
#
# This algorithm can perform well with both high and low noise levels in the original data.
# This particular image has lots of high intensity cosmic ray hits which
# cannot be effectively removed by using the default set of parameters.
# So we reduce ``sigma`` (mapped to astroscrappy's ``sigclip``) from 4.5 to 2 to mark more hits.
# We also reduce ``objlim``, the contrast between the Laplacian image and the fine structured image
# to clean the high intensity bright cosmic ray hits.
# We also modify the ``readnoise`` parameter to obtain better results.

sji_cleaned = sji_frame.remove_cosmic_rays(
    method="astroscrappy",
    sigma=2,
    method_kwargs={"objlim": 2, "readnoise": 4},
)

###############################################################################
# ``remove_cosmic_rays`` returns a cleaned cube with the same metadata and coordinates.
# We now plot the noisy frame and the cleaned frame side-by-side.

fig = plt.figure(figsize=(12, 5), constrained_layout=True)

ax = fig.add_subplot(121, projection=sji_frame.wcs)
sji_frame.plot(axes=ax, vmin=0, vmax=500)
ax.set_title("Original")

ax1 = fig.add_subplot(122, projection=sji_cleaned.wcs)
sji_cleaned.plot(axes=ax1, vmin=0, vmax=500)
ax1.set_title("Cleaned")

ax1.coords[1].set_ticks_visible(False)
ax1.coords[1].set_ticklabel_visible(False)

###############################################################################
# For any cosmic ray removal, it is important to check the results.

plt.show()
