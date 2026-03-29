"""
=====================
Remove IRIS SJI Dust
=====================

In this example we will show how to remove detector dust from an IRIS SJI cube.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch

from irispy.io import read_files
from irispy.utils.dustbuster import clean_sji, clean_sji_regions

###############################################################################
# We start by downloading a small 2832 SJI file used in the crop example.
#
# In this case, we will use ``pooch`` to keep the example self-contained
# but using your browser will also work.

sji_filename = pooch.retrieve(
    "http://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2014/09/19/20140919_051712_3860608353/iris_l2_20140919_051712_3860608353_SJI_2832_t000.fits.gz",
    known_hash="7ec0f3d63d97bc7620675c78fb6c670ef5b4249d31ef7818435b629c04b72f60",
)

###############################################################################
# We can now open the SJI cube.
# To keep the example fast, we only clean the first 5 frames.

sji_2832 = read_files(sji_filename, memmap=False)[:5]

###############################################################################
# We will now compare the standard dust cleanup to the region-based version.
# Both methods use the calibration dust map. The region-based version also
# expands those seed pixels into connected negative regions in the data.

cleaned_default = clean_sji(sji_2832)
cleaned_regions = clean_sji_regions(sji_2832)

change = np.maximum(
    np.abs(cleaned_default.data - sji_2832.data),
    np.abs(cleaned_regions.data - sji_2832.data),
)
frame_idx = int(np.nanargmax(np.nansum(change, axis=(1, 2))))
image_data = (sji_2832.data, cleaned_default.data, cleaned_regions.data)

vmin, vmax = np.nanpercentile(image_data[1][frame_idx], [1, 99])

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
for ax, image, title in zip(
    axes,
    image_data,
    ("Original", "clean_sji", "clean_sji_regions"),
    strict=True,
):
    ax.imshow(image[frame_idx], origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(f"Dust cleanup comparison, frame {frame_idx}")

plt.show()
