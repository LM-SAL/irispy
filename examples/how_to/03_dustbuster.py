"""
=====================
Remove IRIS SJI Dust
=====================

In this example we will show how to remove detector dust from an IRIS SJI cube.
"""

import matplotlib.pyplot as plt
import pooch

from irispy.io import read_files
from irispy.utils import clean_sji_dust, get_sji_dust_params

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
# The dust cleaner also needs a calibration index and the matching bad-pixel map.
# We download the current pair from the IRIS calibration archive.

flat_index_path = pooch.retrieve(
    "https://soho.nascom.nasa.gov/sdb/iris/data/20260326_032515_flat.genx",
    known_hash="40de195c55b0c5e04acb5f6f55883603c74a71bac6a5d639ec73f9d39d076b24",
)
bad_pixel_path = pooch.retrieve(
    "https://soho.nascom.nasa.gov/sdb/iris/data/20260326_032515_badpix.geny",
    known_hash="c4d1884fb1a4f09b6ce4fe150a0aadab2664e479d86ae0ae063d8daa559e230d",
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
# We can now look up the dust-mask parameters and clean the cube.

dust_params = get_sji_dust_params(
    sji_2832,
    flat_index_path=flat_index_path,
    bad_pixel_path=bad_pixel_path,
)
cleaned_sji_2832 = clean_sji_dust(sji_2832, **dust_params)
cleaned_map = cleaned_sji_2832.to_maps(0)

ax = fig.add_subplot(122, projection=cleaned_map.wcs)
cleaned_map.plot(axes=ax, vmin=0, vmax=500)
ax.set_title("Dust cleaned SJI frame")

plt.tight_layout()
plt.show()
