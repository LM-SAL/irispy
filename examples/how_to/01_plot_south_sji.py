"""
================
Plot a quick SJI
================

In this example we will show how to plot a South Pole SJI dataset.
"""

import matplotlib.pyplot as plt
import pooch

from irispy.io import read_files

###############################################################################
# `We start with getting data from the IRIS data archive <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20230211_083601_3880012095_2023-02-11T08%3A36%3A012023-02-11T08%3A36%3A01.xml>`__.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.
#
# You will need to update the path to the data in the next section if you do that.

sji_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2023/02/11/20230211_083601_3880012095/iris_l2_20230211_083601_3880012095_SJI_2832_t000.fits.gz",
    known_hash="282595a2c0a5400ef98d1df0172618483dc74b5b56ff16a0f388723363c6f0f1",
)

###############################################################################
# We will now open the data using a helper function which is designed to read
# all files from a single observation.

sji_2832 = read_files(sji_filename)

###############################################################################
# Printing will give us an overview of the SJI dataset.

print(sji_2832)

###############################################################################
# We will now plot the IRIS SJI data.
#
# You can also change the axis labels and ticks if you so desire.
# `WCSAxes provides us an API we can use. <https://docs.astropy.org/en/stable/visualization/wcsaxes/index.html>`__

# Note that the .get_animation() is used to animate this example and is not required normally.
ax = sji_2832.plot().get_animation()
plt.title(f"IRIS SJI {sji_2832.meta['TWAVE1']}", pad=25)

###############################################################################
# Finally we will output a frame of the SJI into a sunpy Map.

sji_map = sji_2832.to_maps(0)
print(sji_map)

###############################################################################
# We can now plot the SJI Map using sunpy Map's plotting capabilities.

fig = plt.figure()
ax = fig.add_subplot(projection=sji_map.wcs)
sji_map.plot(axes=ax)

plt.show()
