.. _irispy-migration:

*************************************
Migrating to the gWCS raster handling
*************************************

This version rewrites how spectrograph (raster) observations are represented.
Every spectral window is now a single `~irispy.spectrograph.SpectrogramCube`
with a `gWCS <https://gwcs.readthedocs.io/>`__ that describes wavelength, sky
position, time, and raster step (and raster scan for multi-file observations)
together.
This page shows the old and new versions of the most common patterns.

One cube per spectral window
============================

Reading a multi-file raster observation used to return a
``SpectrogramCubeSequence`` per window, indexed by raster scan.
It now returns one 4D cube with the scan number as the leading array axis:

.. code-block:: python

    raster = read_files("iris_l2_..._raster.tar.gz")
    mg_ii = raster["Mg II k 2796"]

    # Old: sequence indexed by raster scan
    first_scan = mg_ii[0]

    # New: one 4D cube (scan, step, slit, wavelength)
    first_scan = mg_ii.raster_slice(0)
    all_scans = mg_ii.split_rasters()

``raster_slice`` and ``split_rasters`` also work on single-file cubes, where
they return the cube itself, so code can treat both cases uniformly.

The ``wcs`` vs ``fits_wcs`` split
=================================

``cube.wcs`` is now a gWCS built from the per-exposure pointing tables in the
FITS AUX extension.
It is the most accurate coordinate description and is what ``crop``, ``plot``,
and ``axis_world_coords`` use.

``cube.fits_wcs`` is the plain linear `astropy.wcs.WCS` built from the FITS
window header, kept for interoperability with code that needs a FITS WCS, for
example reprojection.
It was previously called ``basic_wcs``; that name still works but emits a
`DeprecationWarning` and will be removed in a future release.

The two can differ by a few arcseconds along the scan direction because
``fits_wcs`` is a single linear approximation of the per-step pointing.
On combined multi-file cubes ``fits_wcs`` is `None`; take a single raster
first, for example ``cube.raster_slice(0).fits_wcs``.

For the most common use, building a `~astropy.coordinates.SkyCoord` in the
IRIS pointing frame, you no longer need a WCS at all:

.. code-block:: python

    # Old
    from astropy.wcs.utils import wcs_to_celestial_frame

    iris_frame = wcs_to_celestial_frame(cube.basic_wcs.celestial)

    # New
    iris_frame = cube.celestial_frame
    target = SkyCoord(-338 * u.arcsec, 275 * u.arcsec, frame=iris_frame)

``celestial_frame`` works on both spectrograph and SJI cubes, on combined
multi-file cubes, and on any slice of them.

Cropping
========

The gWCS exposes more world coordinates, so ``crop`` needs one entry per world
object: ``SpectralCoord``, ``SkyCoord``, ``Time``, raster step (and raster
scan for combined cubes).
Old two-element calls raise a ``ValueError`` from ndcube:

.. code-block:: python

    # Old
    cube.crop([SpectralCoord(280, unit=u.nm), target],
              [SpectralCoord(280, unit=u.nm), target])

    # New: one entry per world object, None means "do not crop this one"
    cube.crop([SpectralCoord(280, unit=u.nm), None, None, None],
              [SpectralCoord(280, unit=u.nm), None, None, None])

When cropping around a sky position, the simplest robust pattern is to build
complete world tuples from pixel indices:

.. code-block:: python

    step, slit_pixel, _ = cube.fits_wcs.world_to_array_index(wavelength, target)
    spectrum = cube.crop(
        cube.wcs.array_index_to_world(step, slit_pixel, 0),
        cube.wcs.array_index_to_world(step, slit_pixel, cube.data.shape[-1] - 1),
    )

Time access
===========

Exposure times are now part of the WCS rather than an extra coordinate:

.. code-block:: python

    # Old
    (times,) = cube.axis_world_coords("time", wcs=cube.extra_coords)

    # New
    times = cube.time

On combined multi-file cubes ``cube.time`` is 2D, indexed by
``(raster scan, raster step)``.

Memmap reads
============

Cubes read with ``memmap=True`` now have ``mask=None`` instead of a lazily
computed bad-pixel mask.
Derive it from the unscaled data when needed:

.. code-block:: python

    bad = cube.data == irispy.utils.constants.BAD_PIXEL_VALUE_UNSCALED
