"""
=======================================
Removing Cosmic Ray Hits from IRIS data
=======================================

This example illustrates how to remove cosmic ray hits from a IRIS SJI FITS file.
using `astroscrappy.detect_cosmics <https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html>`__.

``astroscrappy`` is a separate Python package and can be installed separately using ``pip`` or ``conda``.

Currently there is no official routines for removing cosmic ray hits from IRIS data.
This method works decently well for the SJI data but is not effective for the spectrograph data.
Which we will demonstrate here as well.

One other downside is that this method works on single 2D images and does not take advantage of the fact that we have a time series of images.
Which means that we are not using the information about the temporal evolution of the data to identify the cosmic ray hits.
Also that since we have to loop and create a new slice for each image, it can be quite slow to process a long time series of images.
"""

import astroscrappy
import matplotlib.pyplot as plt
import pooch

from astropy.visualization import quantity_support

from sunpy.map import Map

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
raster_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2026/02/09/20260209_215233_3602506433/iris_l2_20260209_215233_3602506433_raster.tar.gz",
    known_hash="bad4a3617d0fd04679203951d7db595df196f79b41a3f6f1f71ce0e301486434",
)

###############################################################################
# We will now open the slit-jaw imager (SJI) file we just downloaded.

sji_2832 = read_files(sji_filename, memmap=False)
raster = read_files(raster_filename, memmap=False)
raster_1403 = raster["Si IV 1403"][10][4]
raster_2796 = raster["Mg II k 2796"][10][4]
del raster

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

mask_sji, clean_data_sji = astroscrappy.detect_cosmics(sji_map.data, sigclip=2, objlim=2, readnoise=4, verbose=False)

###############################################################################
# This returns two variables - ``mask_sji`` is a boolean array depicting whether there is
# a cosmic ray hit at that pixel, ``clean_data_sji`` is the cleaned image after removing those
# hits.
#
# We will need to create a new map with the cleaned data and the original metadata
# and we can now plot the before and after.

clean_sji_map = Map(clean_data_sji, sji_map.meta)

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

###############################################################################
# For comparison, we will now try to remove the cosmic ray hits from the spectrograph data.
# Both to show you the difference in the data but also the individual line profiles.

_, clean_data_raster_1403 = astroscrappy.detect_cosmics(raster_1403.data)
_, clean_data_raster_2796 = astroscrappy.detect_cosmics(raster_2796.data)

###############################################################################
# Currently, since sunpy.Map does not support data with only one celestial axis,
# we cannot use the same plotting method as for the SJI data.

fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

axes[0, 0].imshow(raster_1403.data, aspect="auto", vmin=0, vmax=500, origin="lower")
axes[0, 0].set_title("Original Si IV 1403 Raster")

axes[0, 1].imshow(clean_data_raster_1403, aspect="auto", vmin=0, vmax=500, origin="lower")
axes[0, 1].set_title("Cleaned Si IV 1403 Raster")

axes[1, 0].imshow(raster_2796.data, aspect="auto", vmin=0, vmax=500, origin="lower")
axes[1, 0].set_title("Original Mg II k 2796 Raster")

axes[1, 1].imshow(clean_data_raster_2796, aspect="auto", vmin=0, vmax=500, origin="lower")
axes[1, 1].set_title("Cleaned Mg II k 2796 Raster")

for ax in axes.flat:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

###############################################################################
# Now, finally we will see the effect of the cosmic ray hits on the line profiles
# and how well they are removed.

# We select a particular pixel with a complex line profile
si_iv_idx = 237
mg_ii_idx = 250

si_iv_wave = raster_1403.axis_world_coords("wl")[0].to_value("angstrom")
mg_ii_wave = raster_2796.axis_world_coords("wl")[0].to_value("angstrom")

fig, axes = plt.subplots(2, 1, figsize=(5, 10), constrained_layout=True)
axes[0].plot(si_iv_wave, raster_1403.data[si_iv_idx, :], label="Original")
axes[0].plot(si_iv_wave, clean_data_raster_1403[si_iv_idx, :], label="Cleaned")
axes[0].set_title("Si IV 1403")
axes[0].set_xlabel("Wavelength (Å)")
axes[0].set_xlim(1400, 1406)
axes[0].legend()

axes[1].plot(mg_ii_wave, raster_2796.data[mg_ii_idx, :], label="Original")
axes[1].plot(mg_ii_wave, clean_data_raster_2796[mg_ii_idx, :], label="Cleaned")
axes[1].set_title("Mg II k 2796")
axes[1].set_xlabel("Wavelength (Å)")
axes[1].set_xlim(2792, 2809)
axes[1].legend()

###############################################################################
# While it has done a good job at removing the cosmic ray hits from Mg II k 2796, it has not done a good job at removing the cosmic ray hits from Si IV 1403.
# It is likely that the cosmic ray hits in Si IV 1403 are more complex and have a similar profile
# to the actual line profile, which makes it difficult for the algorithm to distinguish between the
# two. The impression is that it might have removed some of the actual line profile as well, which
# is not ideal.

plt.show()
