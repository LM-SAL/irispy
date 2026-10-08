"""
===================================
Find the fiducial marks on the slit
===================================

This example finds the fiducial marks on the slit in the FUV and NUV spectral windows of a
raster and points them out on spectroheliograms.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch

from irispy.io import read_spectrograph_lvl2
from irispy.utils.fiducials import find_fiducials

###############################################################################
# `We start with a raster of active region 12268 <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20150130_055150_3893010094_2015-01-30T05%3A51%3A502015-01-30T05%3A51%3A50.xml>`__,
# whose three spectral windows cover the FUV1, FUV2 and NUV passbands.
# To keep the download small, we use a cutout of 16 raster steps and 250 rows of the slit,
# which hold the lower of the two fiducial marks.

raster_filename = pooch.retrieve(
    "https://github.com/LM-SAL/irispy-data/releases/download/v1/iris_l2_20150130_055150_3893010094_cutout_raster.fits.gz",
    known_hash="603aa2a5dbe0cf9738e3628451dd24361da05b8eeecd42962ac1253db05888eb",
)
raster = read_spectrograph_lvl2(raster_filename)

###############################################################################
# The fiducial marks are two short gaps in the slit that let no light into the spectrograph,
# so they cross every spectrum as dark rows.
# `~irispy.utils.fiducials.find_fiducials` finds them in the slit profile of a window, the mean
# of the data over raster steps and wavelengths, to a fraction of a pixel. It also returns the
# fraction of the light each mark blocks, a measure of confidence.
#
# A spectroheliogram of each window, here the mean over its wavelengths, shows the mark as a
# dark row across every raster step. The Level 2 pipeline lines the FUV spectra up with the
# NUV, so the mark is on the same row in all three windows.

fig, axes = plt.subplots(1, len(raster), sharey=True, layout="constrained")
for ax, window in zip(axes, raster, strict=True):
    positions, depths = find_fiducials(raster[window])
    print(f"{window}: rows {np.round(positions, 2)}, blocking {np.round(depths, 2)} of the light")
    cube = raster[window][0]
    image = np.ma.masked_array(cube.data, cube.mask).mean(axis=2).T
    vmin, vmax = np.percentile(image.compressed(), [1, 99])
    ax.imshow(image, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
    for position in positions:
        ax.plot(-0.5, position, marker=">", markersize=9, color="C1", clip_on=False)
        ax.plot(image.shape[1] - 0.5, position, marker="<", markersize=9, color="C1", clip_on=False)
    ax.set_title(f"{window}\nmark at row {positions[0]:.2f}")
    ax.set_xlabel("Raster step")
axes[0].set_ylabel("Row along the slit")

plt.show()
