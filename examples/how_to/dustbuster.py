"""
=====================
Remove IRIS SJI Dust
=====================

In this example we will show how to remove detector dust from an IRIS SJI cube.
"""

import matplotlib.pyplot as plt
import pooch

from irispy.io import read_files
from irispy.utils import clean_sji

###############################################################################
# We start by downloading the same SJI file used in the crop example.
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
original_map = sji_2832.to_maps(0)

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection=original_map.wcs)
original_map.plot(axes=ax, vmin=0, vmax=500)
ax.set_title("Original SJI frame")

###############################################################################
# We can now clean the cube.
# The first call will download and cache the pinned calibration files through
# SunPy's data manager.

cleaned_sji_2832 = clean_sji(sji_2832)
cleaned_map = cleaned_sji_2832.to_maps(0)

ax = fig.add_subplot(122, projection=cleaned_map.wcs)
cleaned_map.plot(axes=ax, vmin=0, vmax=500)
ax.set_title("Dust cleaned SJI frame")

plt.tight_layout()
plt.show()
