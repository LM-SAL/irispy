"""
====================
Study umbral flashes
====================

In this tutorial, we are going to work with IRIS data to study umbral flashes :cite:p:`moore1973`.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pooch

import astropy.units as u
from astropy.coordinates import SpectralCoord

from irispy.io import read_files

###############################################################################
# `We start with getting data from the IRIS data archive <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20130902_163935_4000255147_2013-09-02T16%3A39%3A352013-09-02T16%3A39%3A35.xml>`__.
# The full observation is available as a `Level 2 raster tarball <https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2013/09/02/20130902_163935_4000255147/iris_l2_20130902_163935_4000255147_raster.tar.gz>`__.
# To keep the download small, we use a cutout of it that only has the Mg II k 2796 and C II 1336 windows.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.
#
# You will need to update the path to the data in the next section if you do that.

raster_filename = pooch.retrieve(
    "https://github.com/LM-SAL/irispy-data/releases/download/v1/iris_l2_20130902_163935_4000255147_cutout_raster.tar.gz",
    known_hash="b9e55d682f881b6cb7c48e2530bcd3718b42ec6e398cae3d7017b15de3bda6b2",
)
sji_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2013/09/02/20130902_163935_4000255147/iris_l2_20130902_163935_4000255147_SJI_1400_t000.fits.gz",
    known_hash="1f424de4420b729385e81b00df4ba4d868a121486686f17cd1ecdbe7754ee78b",
    # Decompress once: astropy decompresses a .fits.gz file again every time it is opened.
    processor=pooch.Decompress(name="iris_l2_20130902_163935_4000255147_SJI_1400_t000.fits"),
)

###############################################################################
# We will now open the data using a helper function which is designed to read
# all files from a single observation.

# Since this is a large dataset, we will use memory mapping to read the data values
# directly from the FITS files without loading them into memory.

raster = read_files(raster_filename, memmap=True, spectral_windows=["Mg II k 2796", "C II 1336"])
sji_1400 = read_files(sji_filename, memmap=True)

###############################################################################
# We are after the Mg II k and C II lines, which we can select using keys.
# Then we will produce a space-time image of the Mg II k3 line.

mg_ii = raster["Mg II k 2796"][0]
c_ii = raster["C II 1336"][0]

# Instead of using a pixel index, we can crop the data in wavelength space.
lower_corner = [SpectralCoord(279.63, unit=u.nm), None]
upper_corner = [SpectralCoord(279.63, unit=u.nm), None]
mg_crop = mg_ii.crop(lower_corner, upper_corner)

fig = plt.figure()
ax = fig.add_subplot(111, projection=mg_crop.wcs)
# The image is much taller than it is wide, so we let it fill the axes.
mg_crop.plot(axes=ax, aspect="auto")

###############################################################################
# This is a sit-and-stare observation, so the slit stays in almost the same place and
# the vertical axis is really time: the small change in longitude along it is the slit
# following the solar rotation.
#
# The middle section between 60"-75" is on the umbra of a sunspot, even though it is
# not obvious from this image. The umbral oscillations show as a regular pattern of
# dark and bright streaks.
#
# Let us now load the 1400 SJI for context.

plt.figure()
sji_1400[0].plot(vmin=-32000, vmax=-30000)
plt.title("1400 SJI")

###############################################################################
# Slit pixel 220 is on the sunspot's umbra.
# We will compare the k3 intensity (spectral pixel 103 of ``mg_ii``), the
# core of the brightest C II line (spectral pixel 90 of ``c_ii``), and the
# SJI intensity against time (showing the first ~10 minutes only).

# Matplotlib's date formatting works with numpy datetimes, so we convert the times.
mg_ii_times = mg_ii.time[:200].datetime64
c_ii_times = c_ii.time[:200].datetime64

###############################################################################
# The SJI images are typically taken at a different cadence, so we also
# need the corresponding times for the 1400 SJI.
#
# We will take the first 50 to cut down on the size of the data for this example.

times_sji = sji_1400.time[:50].datetime64

###############################################################################
# Now we can plot both spectral lines and SJI for a pixel close to the slit
# at the same Y position (pre-worked out to be row 220 and column 190 of the SJI).

plt.figure()
plt.plot(mg_ii_times, mg_ii.data[:200, 220, 103], label="Mg II k3")
plt.plot(c_ii_times, c_ii.data[:200, 220, 90], label="C II")
(ax,) = plt.plot(times_sji, sji_1400.data[:50, 220, 190], label="1400 SJI")
plt.legend()
plt.ylabel("DN (Memory Mapped Value)")
plt.xlabel("Time (UTC)")
ax.axes.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.axes.xaxis.get_major_locator()))
# Rotates and right-aligns the x labels so they don't crowd each other.
for label in ax.axes.get_xticklabels(which="major"):
    label.set(rotation=30, horizontalalignment="right")

plt.tight_layout()

plt.show()

###############################################################################
# You are now ready to explore all the correlations, anti-correlations,
# and phase differences.

# sphinx_gallery_thumbnail_number = 3
