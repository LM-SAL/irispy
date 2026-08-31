"""
=============================
Make a False-Color RGB Raster
=============================

This example demonstrates how to render an IRIS raster as a false-color image,
where each spatial pixel is colored by the shape of its spectrum.

The spectrum in a pixel is treated as a spectral power distribution and
converted to a single sRGB color, so brightness is the total intensity of the
spectral window and hue is the wavelength within it. Plasma moving towards us
is blue-shifted and appears bluer, plasma moving away appears redder, and a
line with asymmetric wings picks up a color the same line would not have if it
were symmetric.

.. warning::

    This needs the optional ``colorsynth`` dependency, which you can install with
    ``pip install 'irispy-lmsal[rgb]'``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch

import astropy.units as u

from irispy.io import read_files
from irispy.utils.rgb import calculate_rgb

###############################################################################
# `We start with a 64-step raster over a sunspot from the IRIS data archive <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20130902_182935_4000005156_2013-09-02T18%3A29%3A352013-09-02T18%3A29%3A35.xml>`__.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.

raster_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2013/09/02/20130902_182935_4000005156/iris_l2_20130902_182935_4000005156_raster.tar.gz",
    known_hash="91211a52e278fb6e535242d4d6064facf9f93cf24f0a433c276ace1b2d621e7d",
)

###############################################################################
# We will now open the data and select the Si IV 1403 window.

raster = read_files(raster_filename, spectral_windows="Si IV 1403")
si_iv = raster["Si IV 1403"][0]
print(si_iv)

###############################################################################
# The simplest thing we can do is ask the cube's plotter to draw it in false
# color.
#
# Every wavelength in the window is spread across the human visible range, so
# the far wings, which are mostly continuum and noise, get as much of the color
# range as the line itself.

si_iv.plotter.plot_rgb()

plt.show()

###############################################################################
# That washes the line core out, so it is usually worth restricting the color
# range to the wavelengths around the line, and brightening the faint structure
# with a square-root scaling.
#
# ``wavelength_min`` and ``wavelength_max`` set the wavelengths mapped to the
# blue and red ends of the visible range; anything outside them contributes no
# color at all. The colorbar is labelled in both wavelength and Doppler
# velocity, using the ``TWAVE`` rest wavelength from the window metadata, so we
# can pick the range in velocity and convert.

doppler = u.doppler_optical(si_iv.meta.rest_wavelength)
velocity = 100 * u.km / u.s
wavelength_min = (-velocity).to(u.AA, equivalencies=doppler)
wavelength_max = velocity.to(u.AA, equivalencies=doppler)

si_iv.plotter.plot_rgb(
    wavelength_min=wavelength_min,
    wavelength_max=wavelength_max,
    stretch=np.sqrt,
)

plt.show()

###############################################################################
# Not every observation covers ground with both of its non-spectral axes. A
# sit-and-stare revisits the same pointing for hours, so helioprojective
# longitude would show us only the drift from solar rotation. Passing
# ``coordinates="time"`` puts the observation time on the horizontal axis
# instead, which turns the image into a time-distance diagram.

sns_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2021/04/29/20210429_110908_3660259102/iris_l2_20210429_110908_3660259102_raster.tar.gz",
    known_hash="6d07f8dfa4c4644f26dce0c63166d22900d263d555c3d142454ff27fe257688b",
)
sit_and_stare = read_files(sns_filename, spectral_windows="Si IV 1403")["Si IV 1403"][0]

sit_and_stare.plotter.plot_rgb(
    coordinates="time",
    wavelength_min=wavelength_min,
    wavelength_max=wavelength_max,
    stretch=np.sqrt,
)

plt.show()

###############################################################################
# If you want the colors without the plot, for example to composite them with
# another image, `~irispy.utils.rgb.calculate_rgb` returns the RGB array and the
# two dimensional colorbar that goes with it.

rgb, (intensity, wavelength, rgb_colorbar) = calculate_rgb(
    si_iv,
    wavelength_min=wavelength_min,
    wavelength_max=wavelength_max,
    stretch=np.sqrt,
)
print(rgb.shape, rgb.min(), rgb.max())
