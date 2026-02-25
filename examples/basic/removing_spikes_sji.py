"""
===========================================
Removing Cosmic Ray Hits from IRIS SJI data
===========================================

This example illustrates how to remove cosmic ray hits from a IRIS SJI FITS file.
using `astroscrappy.detect_cosmics <https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html>`__.

``astroscrappy`` is a separate Python package and can be installed separately using ``pip`` or ``conda``.

Currently there is no official routines for removing cosmic ray hits from IRIS data.
This method works decently well for the SJI data but is not effective for the spectrograph data.

One other downside is that this method works on single 2D images and does not take advantage of the fact that we have a time series of images.
Which means that we are not using the information about the temporal evolution of the data to identify the cosmic ray hits.
Also that since we have to loop and create a new slice for each image, it can be quite slow to process a long time series of images.
"""

import astroscrappy
import matplotlib.pyplot as plt
import pooch

from sunpy.map import Map

from irispy.io import read_files

###############################################################################
# We start with getting the data.
# This is done by downloading the data from the IRIS archive.
#
# In this case, we will use ``pooch`` as to keep this example self-contained
# but using your browser will also work.
#
# The data is from a BBSO coordination on the 26th February 2026, and is a SJI image in 2832 Å.

sji_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2026/02/06/20260206_210853_3460104433/iris_l2_20260206_210853_3460104433_SJI_2832_t000.fits.gz",
    known_hash="d5088c6a0753ea9ce7b525865ba2edf13a637097ee995985b511833897c88ca6",
)

###############################################################################
# We will now open the slit-jaw imager (SJI) file we just downloaded.

sji_2832 = read_files(sji_filename, memmap=False)

###############################################################################
# Now we will create a sunpy Map from the SJI data.
# We pick the first frame of the SJI data since the cosmic ray hits are present
# at the start of the observation and we want to show the effect of removing them.

sji_map = sji_2832.to_maps(5)

###############################################################################
# Now we will call the `astroscrappy.detect_cosmics <https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html>`__
# to remove the cosmic ray hits.
#
# This algorithm can perform well with both high and low noise levels in the original data.
# The function takes a `~numpy.ndarray` as input so we only pass the map data.
# This particular image has lots of high intensity cosmic ray hits which
# cannot be effectively removed by using the default set of parameters.
# So we reduce ``sigclip``, the Laplacian to noise ratio from 4.5 to 2 to mark more hits.
# We also reduce ``objlim``, the contrast between the Laplacian image and the fine structured image
# to clean the high intensity bright cosmic ray hits.
# We also modify the ``readnoise`` parameter to obtain better results.

mask, clean_data = astroscrappy.detect_cosmics(sji_map.data, sigclip=2, objlim=2, readnoise=4, verbose=False)

###############################################################################
# This returns two variables - mask is a boolean array depicting whether there is
# a cosmic ray hit at that pixel, clean_data is the cleaned image after removing those
# hits.
# We will need to create a new map with the cleaned data and the original metadata
# and we can now plot the before and after.

clean_sji_map = Map(clean_data, sji_map.meta)

fig = plt.figure(figsize=(15, 10))

ax = fig.add_subplot(121, projection=sji_map)
sji_map.plot(axes=ax, vmin=0, vmax=500)
ax.set_title("Original SJI image")

ax1 = fig.add_subplot(122, projection=clean_sji_map)
clean_sji_map.plot(axes=ax1, vmin=0, vmax=500)
ax1.set_title("Cosmic Rays removed")

ax1.coords[1].set_ticks_visible(False)
ax1.coords[1].set_ticklabel_visible(False)
fig.tight_layout()

plt.show()
