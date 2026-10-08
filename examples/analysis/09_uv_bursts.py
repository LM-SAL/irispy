"""
==============
Find UV bursts
==============

UV bursts are compact, short-lived brightening of transition region lines, such as
Si IV, that are often much broader than the surrounding emission.

In this example, we will find them following
:cite:t:`young2018`.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pooch
from matplotlib.colors import LogNorm

import astropy.units as u
from astropy.coordinates import SpectralCoord
from astropy.time import Time

from irispy.io import read_sji_lvl2, read_spectrograph_lvl2
from irispy.utils.bursts import find_si_iv_bursts, find_sji_bursts

###############################################################################
# `We start with getting data from the IRIS data archive <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20130902_182935_4000005156_2013-09-02T18%3A29%3A352013-09-02T18%3A29%3A35.xml>`__.
# The full observation is available as a `Level 2 raster tarball <https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2013/09/02/20130902_182935_4000005156/iris_l2_20130902_182935_4000005156_raster.tar.gz>`__.
# To keep the download small, we use a cutout of it that only has the first raster scan of the Si IV 1403, Mg II k 2796 and C II 1336 windows.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.
#
# You will need to update the path to the data in the next section if you do that.

raster_filename = pooch.retrieve(
    "https://github.com/LM-SAL/irispy-data/releases/download/v1/iris_l2_20130902_182935_4000005156_cutout_raster.tar.gz",
    known_hash="caec6a9d7e4d8fac70163f5f11ef992f3efd5a05b9b5fa6f67b79418bc183e1a",
)
si_iv = read_spectrograph_lvl2(raster_filename, spectral_windows="Si IV 1403")["Si IV 1403"][0]

###############################################################################
# A pixel is part of a burst when its mean Si IV intensity within 50 km/s of the line core
# reaches a threshold. Nothing in this raster reaches the default, 500 DN/s scaled with the
# IRIS sensitivity, so we lower it to 80 DN/s. Like the default, it is for data summed by 2
# in wavelength, so it is halved for this raster. Each row of the table is one event and its
# brightest pixel.

labels, events = find_si_iv_bursts(si_iv, threshold=80)
events

###############################################################################
# ``labels`` holds the event of each pixel, with the spatial WCS of the raster. We outline
# the events on the raster at the Si IV line core.

si_iv_core = SpectralCoord(1402.77 * u.AA)
core_image = si_iv.crop([si_iv_core, None], [si_iv_core, None])

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection=core_image.wcs)
# The raster has 64 steps and 771 slit positions, so we let the image fill the axes.
core_image.plot(axes=ax, plot_axes=["x", "y"], aspect="auto", vmin=0, vmax=100)
# The map is plotted with the raster steps along x, so the labels are transposed to match.
ax.contour(labels.data.T > 0, levels=[0.5], colors="red", linewidths=1.5)
ax.set_title("Bursts in Si IV 1402.77 Å")

###############################################################################
# Next, we reproduce Figure 3 of :cite:t:`young2018`, a burst in 1400 Å slit-jaw images
# taken every 1.7 s by `this sit-and-stare <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20161026_090824_3644103603_2016-10-26T09%3A08%3A242016-10-26T09%3A08%3A24.xml>`__,
# from a cutout of its `Level 2 slit-jaw file <https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2016/10/26/20161026_090824_3644103603/iris_l2_20161026_090824_3644103603_SJI_1400_t000.fits.gz>`__
# with only the area and time range of the figure.

sji_filename = pooch.retrieve(
    "https://github.com/LM-SAL/irispy-data/releases/download/v1/iris_l2_20161026_090824_3644103603_cutout_SJI_1400.fits.gz",
    known_hash="0fc9a8431347b22c00b61f5c1bf665ab94517878e251c2d36e2702be3206aacd",
)
sji_1400 = read_sji_lvl2(sji_filename)

###############################################################################
# Here, a pixel is part of a burst when it is at least 10 standard deviations (``sigma_factor``)
# above the median of its frame. The burst of Figure 3 is the brightest event at 09:53:21.

sji_labels, sji_events = find_sji_bursts(sji_1400)
frame = np.argmin(np.abs(sji_1400.time - Time("2016-10-26T09:53:21")))
in_frame = sji_events[sji_events["frame"] == frame]
burst = in_frame[np.argmax(in_frame["intensity"])]

###############################################################################
# The light curve is the mean of a 7 x 8 pixel box centered on the burst, divided by the median
# of the area, and the inset shows its flickering around the peak. Its shape and timing match
# the paper, but it peaks at about 19 times the median rather than 25, most likely because the
# data were reprocessed in 2025.

y, x = burst["y"], burst["x"]
data = np.where(sji_1400.mask, np.nan, sji_1400.data)
light_curve = np.nanmean(data[:, y - 4 : y + 4, x - 3 : x + 4], axis=(1, 2)) / np.nanmedian(data, axis=(1, 2))
times = sji_1400.time.datetime64

fig = plt.figure(figsize=(11, 4.5), layout="constrained")
ax_image = fig.add_subplot(1, 2, 1, projection=sji_1400[frame].wcs)
# Pixels at or below 0 DN have no logarithm and show the background
ax_image.set_facecolor("black")
sji_1400[frame].plot(axes=ax_image, norm=LogNorm(vmin=5, vmax=burst["intensity"].value))
ax_image.contour(sji_labels.data[frame] == burst["label"], levels=[0.5], colors="blue", linewidths=1.5)
ax_image.set_title(f"1400 Å slit-jaw at {sji_1400.time[frame].isot[11:19]}")

ax_curve = fig.add_subplot(1, 2, 2)
ax_curve.plot(times, light_curve, color="black")
ax_curve.set_xlabel("Time (UTC)")
ax_curve.set_ylabel("Intensity / median")
ax_curve.set_ylim(bottom=0)
ax_curve.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_curve.xaxis.get_major_locator()))
ax_curve.set_title("Burst light curve")
peak = (times >= np.datetime64("2016-10-26T09:52:10")) & (times <= np.datetime64("2016-10-26T09:54:00"))
inset = ax_curve.inset_axes([0.55, 0.45, 0.42, 0.45], xticks=[], yticks=[])
inset.plot(times[peak], light_curve[peak], color="black")
ax_curve.indicate_inset_zoom(inset, edgecolor="blue")

plt.show()
