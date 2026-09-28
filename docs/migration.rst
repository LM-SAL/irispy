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

``raster_slice(0)`` and ``split_rasters`` also work on single-file cubes, where
they return the cube itself (as a one-element tuple for ``split_rasters``), so
code can treat both cases uniformly.

.. warning::

    Integer indexing now follows the array axes of the cube.
    On a single-file cube ``cube[0]`` is raster step 0, a 2D (slit, wavelength)
    image, not the whole raster as it was with a sequence.
    On a combined cube ``cube[0]`` is scan 0.
    Use ``cube.raster_slice(0)`` for "the first raster" in both cases.

The ``wcs`` vs ``fits_wcs`` split
=================================

``cube.wcs`` is now a gWCS built from the per-exposure pointing tables in the
FITS AUX extension.
It is the most accurate coordinate description and is what ``crop``, ``plot``,
and ``axis_world_coords`` use.

``cube.fits_wcs`` is an `astropy.wcs.WCS` that stores the same per-exposure
pointing as a FITS ``-TAB`` lookup table. It is kept for interoperability with
code that needs a FITS WCS, for example reprojection.
``SJICube.basic_wcs`` has been removed; use ``fits_wcs``, which now also exists
on spectrograph cubes. ``SJICube.scaled`` has also been removed; use
``cube.meta["scaled"]``.

The two agree to well under a milliarcsecond on the sky.
On combined multi-file cubes ``fits_wcs`` is `None`; take a single raster
first, for example ``cube.raster_slice(0).fits_wcs``.
Each raster's ``fits_wcs`` uses its own file's observer and date, while the
celestial frame of a combined cube uses the first file's for every raster, so
frame-aware transforms between the two differ by the observer's motion between
files (about 0.15 arcseconds for rasters 15 minutes apart).

For the most common use, building a `~astropy.coordinates.SkyCoord` in the
IRIS pointing frame, you no longer need a WCS at all:

.. code-block:: python

    # Old
    from astropy.wcs.utils import wcs_to_celestial_frame

    iris_frame = wcs_to_celestial_frame(cube.basic_wcs[0].celestial)

    # New
    iris_frame = cube.celestial_frame
    target = SkyCoord(-338 * u.arcsec, 275 * u.arcsec, frame=iris_frame)

``celestial_frame`` works on both spectrograph and SJI cubes, on combined
multi-file cubes, and on any slice of them.

Times
=====

Exposure times are part of the gWCS, so use ``cube.time`` instead of reading
them from the extra coordinates. On a combined cube it has shape (scan, step).

.. code-block:: python

    # Old
    times = cube.axis_world_coords("time", wcs=cube.extra_coords)[0]

    # New
    times = cube.time

Cubes from a combined multi-file read, including the rasters returned by
``raster_slice`` and ``split_rasters``, have no ``time`` extra coordinate.

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

Sky, time, and step coordinates can also be supplied on their own. For example,
to crop a single raster to a sky region or a time interval:

.. code-block:: python

    region = cube.crop([None, bottom_left, None, None],
                       [None, top_right, None, None])
    interval = cube.crop([None, None, start_time, None],
                         [None, None, end_time, None])

For combined rasters, append a fifth entry for the scan coordinate. Leaving it
as ``None`` includes every matching raster. Supplying a scan coordinate selects
that raster. The same partial-coordinate behavior is available through
``crop_by_values``; supply both numeric celestial components for a sky crop.

Partial time crops select exposures whose timestamps fall within the inclusive
time interval, including repeated or nonmonotonic timestamps. Sky crops use the
pointing transform of each exposure to bound the supplied sky region in pixels.
The finite slit width is respected, including for sit-and-stare observations.
All coordinate constraints are applied together.
Crops that leave out some world coordinates need an ndcube version with the
crop-bounds extension point, which is not in a release yet; with a released
ndcube, supply every world coordinate.

The result is one rectangular slice containing the matches, so it can include
intervening pixels or rasters that do not themselves match. A crop with no
matching pixels raises ``ValueError``. Use ``keepdims=True`` to retain axes of
length one.

Memmap reads
============

As before, cubes read with ``memmap=True`` have ``mask=None``.
Derive the bad-pixel mask from the unscaled data when needed:

.. code-block:: python

    bad = cube.data == irispy.utils.constants.BAD_PIXEL_VALUE_UNSCALED

A multi-file raster read with ``memmap=True`` is one dask-backed cube that reads
each file on demand, which works only when all files have the same number of
raster steps.
If the files are ragged, use ``memmap=False`` so shorter rasters can be padded
with masked NaNs.
