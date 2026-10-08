.. _irispy-tutorial-data-idiosyncrasies:

*******************
Data Idiosyncrasies
*******************

This page describes some of the particularities of IRIS data that users should be aware of when working with the data.

Background in FUV data
======================

FUV spectra with longer exposure times show a faint background most likely caused by a light leak from wavelengths significantly longer than the FUV.
This means that the light leak is absorbed at a different CCD depth than the FUV light and thus does not show the same CCD flat-field (which for the FUV is quite prominent and dominated by the CCD annealing pattern).
The light leak effectively acts as an extra "dark current" although it appears to have varying intensity levels for different pointings on the Sun.
This background has been characterized and is automatically removed by ``iris_prep.pro``, and therefore subtracted in level 1.5 and level 2 data.

A background or continuum left under a line can be removed with `~irispy.utils.spectrograph.subtract_background`.
The windows are your choice: take them from the spectra of your observation, clear of the lines, their wings and any weaker lines.
For example, for the C II 1334.53 and 1335.71 Å lines the windows could be 150 to 300 km/s beyond each line.

The plot compares one spectrum from the first raster step before and after a straight-line background is subtracted.
The shaded regions on either side of the C II lines are the background fit windows.
Only unmasked, finite samples in these windows determine the straight-line background.
The fitted background is then evaluated and subtracted at every wavelength, including the C II lines.

.. plot::
    :include-source:

    import matplotlib.pyplot as plt

    import astropy.units as u

    import irispy.data.sample as sample_data
    from irispy.io import read_files
    from irispy.utils.spectrograph import subtract_background

    blue = ([-300, -150] * u.km / u.s).to(u.AA, equivalencies=u.doppler_optical(1334.53 * u.AA))
    red = ([150, 300] * u.km / u.s).to(u.AA, equivalencies=u.doppler_optical(1335.71 * u.AA))
    windows = u.Quantity([blue, red])

    c_ii = read_files(sample_data.RASTER_FITS, spectral_windows="C II 1336")["C II 1336"][0]
    spectrum = c_ii[0, 150]
    corrected = subtract_background(spectrum, windows)

    wavelengths = spectrum.spectral_axis.to_value(u.AA)
    shown = (wavelengths >= blue[0].value) & (wavelengths <= red[1].value)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(wavelengths[shown], spectrum.data[shown], label="Original")
    ax.plot(wavelengths[shown], corrected.data[shown], label="Background subtracted")
    for low, high in windows.to_value(u.AA):
        ax.axvspan(low, high, color="gray", alpha=0.2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set(xlabel="Wavelength [Å]", ylabel=f"Intensity [{spectrum.unit}]")
    ax.legend()
    plt.show()

Coalignment between channels and SJI & spectra
==============================================

In level 2 data the slit-jaw images from different filters and detectors are automatically co-aligned.
This automatic approach is not failsafe, and for precise analysis one should always check if they match.
There are two spectral marks on the slit that are called fiducials and block the light from entering.
They are used for calibration, and their position should match between slit-jaw images.
With smaller fields of view only one of the fiducials is visible.

.. note::

    The position of the slit in different slit-jaw channels is not necessarily the same.
    Depending on the observing program, different slit-jaw filters may be exposed at different parts of a raster.
    This is particularly true for two or four step rasters.
    In such cases the alignment should have in mind the header coordinates from ``CRPIX`` and ``CRVAL``.

.. figure:: ../_static/images/tutorial/sji_fiducials.jpg
    :align: center

    Position of fiducial marks on a slit-jaw image.

As in the slit-jaw images, so too the NUV and FUV spectrograms are co-aligned in level 2 data.
These too should be checked for the alignment, both between FUV, NUV and slit-jaws.
In spectrograms the fiducial marks appear as solid black lines along the wavelength direction, and they should appear in the same exact spatial position for the NUV and FUV channels.

.. figure:: ../_static/images/tutorial/sp_fiducials.jpg
    :align: center

    Position of fiducial marks on an NUV spectrogram.

The marks are two gaps in the slit, each two pixels (0.33″) long, one in each half of the CCD :cite:p:`depontieu2014`, and the Level 2 pipeline shifts the spectra so that the marks fall on the same rows in every spectral window :cite:p:`wulser2018`.
In the ten observations from 2013 to 2026 we checked, the marks are 89.7″ apart, at rows 238.0 and 777.1 (counting from 0) of a window that covers the whole slit without spatial summing.
They are clear in the NUV windows but often not visible in the FUV windows: only three of the ten observations show them there (2014-07-08, 2015-01-30 and 2021-04-29).
Some FUV windows also show a dark row that is not a mark, about 61 rows (10″) below the lower mark.

`irispy.utils.fiducials.find_fiducials` finds the marks in a spectral window; see :ref:`sphx_glr_generated_gallery_how_to_07_find_fiducials.py`.
In the observations we checked, the FUV and NUV windows line up to a fraction of a pixel; see :ref:`irspy_known_issues`.

Cosmic rays
===========

IRIS passes through the South Atlantic Anomaly (SAA) on a regular basis.
The impact of energetic particles on the CCD camera causes bright hits/pixels.
These can be removed with any of the multitude of cosmic ray removal procedures available in Python.
``irispy`` includes ``remove_cosmic_rays`` methods on SJI and raster cubes.
These methods support sliding sigma clipping via `rsliding <https://git.ias.u-psud.fr/avoyeux/rsliding>`__ and `astroscrappy <https://astroscrappy.readthedocs.io/>`__.
There are two examples: :ref:`sphx_glr_generated_gallery_calibration_01_remove_spikes_sg.py` showcases how to remove spikes from IRIS spectrogram data, and :ref:`sphx_glr_generated_gallery_calibration_02_remove_spikes_sji.py` details how to remove spikes from IRIS SJI data.
Cosmic rays are **not removed** from the IRIS data during normal calibration/pipeline processing to avoid introducing artifacts.

Particles on slit-jaw images
============================

The slit-jaw CCD contains some particles that cause dark regions of order up to a few arcseconds in size in the slit.
These features are marked as bad pixels and set to zero values (0) in ``iris_prep.pro`` so they can be easily recognized during data analysis.
The particles are stable in position and do not let any light through - they are completely dark.
They are most prominent in the FUV images (1400 Å and 1330 Å) and much less visible in the NUV images (2796 Å and 2830 Å).

CCD camera readout noise
========================

When both spectrograph cameras are read out simultaneously, a read interference noise pattern is superimposed on the resulting data, which can impact the weakest lines in the FUV.
The readout noise is only present when the last two digits of the OBSID are less than 50 (for the OBSID generations starting with 38,40 or 41 numbers).
This can be avoided altogether by reading the cameras sequentially, and most of the data is now observed using the sequential read (last two digits in OBSID larger than 50 for OBSID).
For OBSIDs starting with 36, those whose 4th digit are 0-4 are sequential read and those with 5-9 are simultaneous readout (3624103603 is sequential and 3629103603 is simultaneous).
Also the high-cadence flare OBSIDS 4204700126-4204700143 are simultaneous readout.

Flagging of saturated data
==========================

Some observations show strong solar activity and resulting saturation either on the CCD or (especially in OBS sequences where data is summed) in the A/D converter.
Level 2 processing flags saturated pixels as ``inf``, but level 2 files hold none: every sample, saturated or merely bright, is clipped at 16182 DN (``irispy.utils.constants.SATURATION_LIMIT``), and their ``NSATPIX`` and ``TSATPXn`` keywords are 0.
Pass it as ``saturation_limit`` to `~irispy.utils.moments.calculate_moments` or `~irispy.utils.mg_features.calculate_mg_features` to leave such pixels out of moment or Mg II feature maps.

Cosmetic finishing in quicklook movies
======================================

The quicklook movies on the IRIS website use a standard set of color tables and intensity scales that have been designed to give a consistent, recognizable and generally pleasing appearance to IRIS observations of a range of solar features.
For users who wish to replicate the appearance of off-the-shelf IRIS movies with their own processed data, you will need to do some work.

Color Tables
------------

The color tables are stored within `sunpy` and can be loaded with the `sunpy.visualization.colormaps` module.
The IRIS color tables are named 'irissji1330', 'irissji1400', 'irissji1600', 'irissji2796', 'irissji2832', 'irissji5000', 'irissjiFUV', 'irissjiNUV' and 'irissjiSJI_NUV'.
Reddish color tables are defined for the IRIS FUV slit-jaw images and spectra (the 1330 Å SJI channel uses a more yellowish-red than the 1400); yellowish tables are defined for the NUV images and spectra.

The color tables can be used in plotting routines, for example:

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> # Register the IRIS color tables with matplotlib
    >>> import sunpy.visualization.colormaps

    >>> cmap = plt.get_cmap('irissji1330')
    >>> cmap
    <matplotlib.colors.LinearSegmentedColormap object at ...>

This is used by default for plotting IRIS slit-jaw images with ``irispy`` plotting routines, but can be used in any plotting routine that accepts a colormap.

Intensity Scaling
-----------------

The IRIS data consists of 14-bit pixel values (DN in level 0 data can range from 0-16383; corrections applied during processing to level 2 can move the data values somewhat outside this range).

There is no Python code to automatically determine the best scaling for a given observation, but the `irispy.utils.image_clipping` routine can be used to determine good values for ``vmin`` and ``vmax`` for scaling the data in a plot.

Cleaning Up
-----------

The slit-jaw CCD contains some particles that cause dark regions of order up to a few arcseconds in size in the slit.
These features are marked as bad pixels and set to zero values (0) in ``iris_prep.pro`` so they can be easily recognized during data analysis.
The particles are stable in position and do not let any light through - they are completely dark.
They are most prominent in the FUV images (1400 Å and 1330 Å) and much less visible in the NUV images (2796 Å and 2830 Å).

These dust spots can be cosmetically corrected in slit-jaw images and movies but there is currently no routine to do this in ``irispy``.
