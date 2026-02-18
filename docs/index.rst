.. _irispy-index:

************************
``irispy`` documentation
************************

``irispy`` is an open-source Python package that provides tools to read, manipulate, and visualize `Interface Region Imaging Spectrograph (IRIS) <https://iris.lmsal.com/>`__ data.
`The data is publicly available and provides access to co-aligned SDO/AIA data and more. <https://iris.lmsal.com/search/>`__

The goal of ``irispy`` is to provide a set of classes for handling both imaging (slit-jaw) and spectral observations (spectrograph).
The classes link the observations with various forms of supporting data including: measurement uncertainties; units; a data mask to mark pixels with unreliable or unphysical data values; WCS (World Coordinate System) transformations that describe the position, wavelengths, and times represented by the pixels; and general metadata.
These classes also provide methods for applying a number of calibration routines including exposure time correction and conversion between data number, photons, and energy units, referred to as radiometric calibration.

.. warning::

    Please be aware that the package name on pypi and conda-forge is ``irispy-lmsal`` to avoid name clashes with other packages.
    However, the package is imported as ``irispy`` and is referred to as ``irispy`` in the documentation.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card::
        :class-card: card

        Getting started
        ^^^^^^^^^^^^^^^
        .. toctree::
          :maxdepth: 1

          iris
          tutorial/index
          generated/gallery/index

    .. grid-item-card::
        :class-card: card

        Other info
        ^^^^^^^^^^
        .. toctree::
          :maxdepth: 1

          known_issues
          contributing
          reference/index
          changelog

.. _Interface Region Imaging Spectrograph: https://iris.lmsal.com/

Getting help
============

If you would like to get in touch with someone who works on ``irispy`` **for any reason**, we suggest opening an issue on the `irispy GitHub issue tracker <https://github.com/LM-SAL/irispy/issues>`__.
