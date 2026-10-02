"""
=============
Crop IRIS SJI
=============

In this example we will show how to crop an IRIS dataset, and a particularity of the crop
operation.
"""

import matplotlib.pyplot as plt
import pooch

import astropy.units as u
from astropy.coordinates import SkyCoord

from irispy.io import read_files
from irispy.obsid import ObsID

###############################################################################
# `We start with getting data from the IRIS data archive <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20140919_051712_3860608353_2014-09-19T05%3A17%3A122014-09-19T05%3A17%3A12.xml>`__.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.
#
# You will need to update the path to the data in the next section if you do that.

sji_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2014/09/19/20140919_051712_3860608353/iris_l2_20140919_051712_3860608353_SJI_2832_t000.fits.gz",
    known_hash="7ec0f3d63d97bc7620675c78fb6c670ef5b4249d31ef7818435b629c04b72f60",
)

###############################################################################
# We will now open the data using a helper function which is designed to read
# all files from a single observation.

sji_2832 = read_files(sji_filename)

###############################################################################
# Printing will give us an overview of the file.

print(sji_2832)
# ``.meta`` holds the full primary FITS header, which is too long to print here.

###############################################################################
# Can't remember what OBSID 3860608353 is? ``irispy.obsid.ObsID`` describes it.

print(ObsID(sji_2832.meta["OBSID"]))

###############################################################################
# Now, we will plot the SJI. By default, **irispy** will
# color the spatial axes.

# ``get_animation`` lets Sphinx Gallery render the sequence as an animation.
animation = sji_2832.plot().get_animation()

###############################################################################
# We also have the option of going directly to an individual frame.

sji_45 = sji_2832[45]
print(sji_45)

###############################################################################
# The cutout below needs the coordinate frame of the IRIS data, which the cube
# provides as ``celestial_frame``.

sji_frame = sji_45.celestial_frame
bbox = [
    SkyCoord(-750 * u.arcsec, 90 * u.arcsec, frame=sji_frame),
    SkyCoord(-750 * u.arcsec, 95 * u.arcsec, frame=sji_frame),
    SkyCoord(-700 * u.arcsec, 90 * u.arcsec, frame=sji_frame),
    SkyCoord(-700 * u.arcsec, 95 * u.arcsec, frame=sji_frame),
]

###############################################################################
# This observation has a 45 degree roll. The image is not rotated because plotting
# shows the data as they are stored in the file, so we add a coordinate grid to make
# the roll clear.
#
# Now, let us cut out the top sunspot. ``crop`` takes the corners of the region as
# ``SkyCoord`` objects (longitude first, then latitude) and returns the smallest pixel
# box that contains them all. As the data are rotated with respect to the solar
# north-south frame, we give all four corners, and the result is larger than the box
# they outline, as the overlaid points show.

sji_cutout = sji_45.crop(*bbox)

plt.figure()
ax = sji_cutout.plot()
# Plot each corner of the box
[ax.plot_coord(coord, "o") for coord in bbox]
# WCSAxes needs ``grid_type="contours"`` to draw the grid of this WCS correctly.
ax.coords.grid(grid_type="contours")

plt.show()

# sphinx_gallery_thumbnail_number = 2
