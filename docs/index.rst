************************
``irispy`` documentation
************************

``irispy`` is a Python package designed for reading, manipulating, and visualizing data taken with the `Interface Region Imaging Spectrograph`_ (IRIS).

What is IRIS?
=============

IRIS is a satellite-borne solar scanning slit spectrograph, funded as part of NASA's Small Explorer program and launched in June 2013.
It provides simultaneous UV images and spectra of the regions between the visible surface of the Sun and its outer atmosphere (photosphere, chromosphere, transition region, and corona) with 0.33-0.4 arcsec spatial resolution, two-second temporal resolution and 1 km/s velocity resolution over a field-of-view of up to 175 arcsec x 175 arcsec.
IRIS images are provided by its Slit-Jaw Imager (SJI) in four passbands: C II 1330 Å, Si IV 1400 Å, Mg II K 2796 Å and Mg II wing 2830 Å.
If you desire more detail, the `instrument paper`_ PDF provides this.

As a scanning slit spectrograph, IRIS disperses sunlight by passing it through a narrow slit and onto a CCD.
The spectrograph can operate in two basic modes: sit-and-stare, where the slit is aligned with a single position on the Sun; and raster, where the slit moves sequentially across the Sun perpendicular to the long axis of the slit in a predetermined number of steps and step size.
When the last position is reached, the slit returns to the origin position and starts again.
IRIS has many different observing programs involving different numbers of raster steps, step sizes, exposure times, etc., making it a powerful and flexible tool for solar physics.

The complexity of IRIS leads to a variety of different data products with different dimensionalities and ways in which they are used by scientists.
Therefore, a powerful yet flexible suite of data analysis tools is required to enable users to efficiently, reliably, and effectively pursue their science goals.
This is the aim of irispy.

`The data is publicly available and provides access to co-aligned SDO/AIA data and more. <https://iris.lmsal.com/search/>`__

What is ``irispy``?
====================

``irispy`` is an open-source Python package that provides tools to read, manipulate, and visualize IRIS data.
It provides a set of classes for handling both SJI and spectrograph observations.
These link the observations with various forms of supporting data including: measurement uncertainties; units; a data mask to mark pixels with unreliable or unphysical data values; WCS (World Coordinate System) transformations that describe the position, wavelengths, and times represented by the pixels; and general metadata.
These classes also provide methods for applying a number of calibration routines including exposure time correction and conversion between data number, photons, and energy units.

.. warning::

   Please be aware that the package name on pypi and conda-forge is ``irispy-lmsal`` to avoid name clashes with other packages.
   However, the package is imported as ``irispy`` and is referred to as ``irispy`` in the documentation.

Getting help
============

If you would like to get in touch with someone who works on ``irispy`` **for any reason**, we suggest opening an issue on the `irispy GitHub issue tracker <https://github.com/LM-SAL/irispy/issues>`__.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card::
        :class-card: card

        Getting started
        ^^^^^^^^^^^^^^^
        .. toctree::
          :maxdepth: 1

          installation
          guide
          generated/gallery/index
          known_issues

    .. grid-item-card::
        :class-card: card

        Other info
        ^^^^^^^^^^
        .. toctree::
          :maxdepth: 1

          contributing
          reference/index
          changelog

.. _Interface Region Imaging Spectrograph: https://iris.lmsal.com/
.. _instrument paper: https://www.lmsal.com/iris_science/doc?cmd=dcur&proj_num=IS0196&file_type=pdf
