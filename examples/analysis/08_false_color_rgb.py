"""
=============================
Make a False-Color RGB Raster
=============================

In this example, we are going to render each spectrum as one RGB pixel.

The spectrum in a pixel is treated as a spectral power distribution and
converted to a single sRGB color, so brightness is the total intensity of the
spectral window and hue is the wavelength within it. Plasma moving towards us
is blue-shifted and appears bluer, plasma moving away appears redder, and a
line with asymmetric wings picks up a color the same line would not have if
it were symmetric.

.. warning::

    This needs the optional ``colorsynth`` dependency, which you can install
    with ``pip install 'irispy-lmsal[rgb]'``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch

import astropy.units as u

from irispy.io import read_files

###############################################################################
# We will start by getting some data from the IRIS archive.
#
# Using the url: https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20130902_182935_4000005156_2013-09-02T18%3A29%3A352013-09-02T18%3A29%3A35.xml
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.
#
# You will need to update the path to the data in the next section if you do that.

raster_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2013/09/02/20130902_182935_4000005156/iris_l2_20130902_182935_4000005156_raster.tar.gz",
    known_hash="91211a52e278fb6e535242d4d6064facf9f93cf24f0a433c276ace1b2d621e7d",
)

# We will only focus on the Si IV.
si_iv = read_files(raster_filename, spectral_windows="Si IV 1403")["Si IV 1403"][0]

###############################################################################
# By default, the metadata stored in the cube will be used and that means,
# it will use the rest wavelength, the default maps +/-100 km/s to color
# and uses an asinh transform with a 25 km/s scale to emphasize small shifts.

si_iv.plotter.plot_rgb()

###############################################################################
# If you want to override the range and use linear wavelength mapping,
# you can use a square-root intensity stretch which brightens faint signal
# and  wavelengths outside the limits contribute no color.

doppler = u.doppler_optical(si_iv.meta.rest_wavelength)
wavelength_min, wavelength_max = ([-200, 200] * u.km / u.s).to(u.AA, equivalencies=doppler)
si_iv.plotter.plot_rgb(
    wavelength_min=wavelength_min, wavelength_max=wavelength_max, wavelength_norm=None, stretch=np.sqrt
)

###############################################################################
# For a sit-and-stare, you can use ``coordinates="time"`` to plot time along
# the horizontal axis.

sns_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2021/04/29/20210429_110908_3660259102/iris_l2_20210429_110908_3660259102_raster.tar.gz",
    known_hash="6d07f8dfa4c4644f26dce0c63166d22900d263d555c3d142454ff27fe257688b",
)
sit_and_stare = read_files(sns_filename, spectral_windows="Si IV 1403")["Si IV 1403"][0]

sit_and_stare.plotter.plot_rgb(coordinates="time")

plt.show()
