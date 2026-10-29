.. _irispy-migration:

*************************************
Migrating to the gWCS raster handling
*************************************

This version rewrites how spectrograph (raster) observations are represented.
Every spectral window is now a single 4D `~irispy.spectrograph.SpectrogramCube`
with a `gWCS <https://gwcs.readthedocs.io/>`__ that describes wavelength, sky
position, time, raster step and raster scan together.
This page shows the old and new versions of the most common patterns.

One cube per spectral window
============================

Reading raster files used to return a ``SpectrogramCubeSequence`` per window,
indexed by raster scan, with one raster for each file.
It now returns one 4D cube per window, with the raster scan as the leading array
axis, so ``[k]`` is still raster ``k``, a 3D (step, slit, wavelength) cube, and
iterating over the cube gives each raster. One raster file, such as a
sit-and-stare, gives a cube with one raster:

.. code-block:: python

    raster = read_files("iris_l2_..._raster.tar.gz")
    mg_ii = raster["Mg II k 2796"]  # one 4D cube (raster, step, slit, wavelength)

    first_raster = mg_ii[0]  # as before
    all_rasters = list(mg_ii)

The cube has the header metadata of its first file, with the ``DATE_END`` of its
last. The per-exposure metadata, such as ``cube.meta["exposure time"]``, and
``cube.time`` have shape (raster, step).
``cube.meta["saturated"]``, if there are a few saturated samples, holds only their positions:
``np.asarray(cube.meta["saturated"])`` gives the `True`/`False` array, and it cannot be changed in place.

When `~irispy.io.read_files` reads several products, such as a downloaded
observation folder, the `~irispy.spectrograph.RasterCollection` of the rasters is
keyed ``"raster"``, not by its first spectral window:

.. code-block:: python

    files = read_files("20211001_060925_3683602040/")

    # Old
    mg_ii = files["C II 1336"]["Mg II k 2796"]

    # New
    mg_ii = files["raster"]["Mg II k 2796"]

The ``wcs`` vs ``fits_wcs`` split
=================================

``cube.wcs`` is now a gWCS built from the per-exposure pointing tables in the
FITS AUX extension.
It is the most accurate coordinate description and is what ``crop``, ``plot``,
and ``axis_world_coords`` use.

``cube.fits_wcs`` is an `astropy.wcs.WCS` that stores the same per-exposure
pointing as a FITS ``-TAB`` lookup table, the WCS that ``cube.wcs`` used to be.
It is kept for interoperability with code that needs a FITS WCS, for example
reprojection, as ``SJICube.fits_wcs`` is for slit-jaw cubes.

The two agree to well under a milliarcsecond on the sky.
The ``fits_wcs`` of the cube of a window is a list with one FITS WCS per raster,
as ``SJICube.fits_wcs`` has one per frame, and ``cube.fits_wcs[k]`` is that of
raster ``k``, as is ``cube[k].fits_wcs``.
Each raster's FITS WCS uses its own file's observer and date, while the
celestial frame of the cube uses the first file's for every raster, so
frame-aware transforms between the two differ by the observer's motion between
files (about 0.15 arcseconds for rasters 15 minutes apart).

World coordinates
=================

``cube.wcs.pixel_to_world`` now returns a ``SpectralCoord``, a ``SkyCoord``, a
``Time``, the raster step and the raster scan (no scan for one raster,
``cube[k]``), and ``world_to_pixel`` needs all of them. The wavelengths are in nm, where they were
in m, and the longitude and latitude in arcsec, where they were in deg, so the
``.value`` of a coordinate and ``axis_world_coords_values`` change with them.

To find the pixel of a sky position, ask the FITS WCS of one raster, which needs
only a wavelength and the position:

.. code-block:: python

    target = SkyCoord(-338 * u.arcsec, 275 * u.arcsec, frame=cube.celestial_frame)
    wavelength = SpectralCoord(279.63, unit=u.nm)
    step, slit_pixel, _ = cube[0].fits_wcs.world_to_array_index(wavelength, target)

For the most common use, building a `~astropy.coordinates.SkyCoord` in the
IRIS pointing frame, you no longer need a WCS at all:

.. code-block:: python

    # Old
    from astropy.wcs.utils import wcs_to_celestial_frame

    iris_frame = wcs_to_celestial_frame(cube.wcs.celestial)

    # New
    iris_frame = cube.celestial_frame
    target = SkyCoord(-338 * u.arcsec, 275 * u.arcsec, frame=iris_frame)

``celestial_frame`` works on both spectrograph and SJI cubes, and on any slice of
them.

Plots and coordinate order
==========================

The gWCS lists its world coordinates as wavelength, longitude, latitude, time,
raster step and raster scan. The FITS WCS listed latitude
before longitude, so code that picked WCSAxes coordinates by position, for
example ``ax.coords[0]`` for latitude, now gets longitude. Pick them by name:

.. code-block:: python

    # Old
    ax.coords[0].set_ticklabel(fontsize=8)  # latitude

    # New
    ax.coords["custom:pos.helioprojective.lat"].set_ticklabel(fontsize=8)

``plot`` puts latitude on the edge of the slit axis and longitude on the edge of
the raster step axis, so there is no need to set their positions yourself. A
raster map at one wavelength is drawn with its raster steps along x, like a map
of the Sun. The time and raster step coordinates are hidden unless you ask for them with
``axes_coordinates``, for example
``cube.plot(plot_axes=["x", "y"], axes_coordinates=["custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat", "time"])``
puts the time on the top edge of an image with the raster steps along x.

Times
=====

The times of the exposures are part of the gWCS, so use ``cube.time`` instead of reading
them from the extra coordinates. It has shape (raster, step).

.. code-block:: python

    # Old
    times = cube.axis_world_coords("time", wcs=cube.extra_coords)[0]

    # New
    times = cube.time

Cropping
========

The gWCS exposes more world coordinates, so ``crop`` needs one entry per world
object. Old two-element calls raise a ``ValueError`` from ndcube:

.. code-block:: python

    # Old
    cube.crop([SpectralCoord(280, unit=u.nm), target],
              [SpectralCoord(280, unit=u.nm), target])

    # New, on one raster
    cube[0].crop([SpectralCoord(280, unit=u.nm), None, None, None],
                 [SpectralCoord(280, unit=u.nm), None, None, None])

    # New, on the cube of a whole window
    cube.crop([SpectralCoord(280, unit=u.nm), None, None, None, None],
              [SpectralCoord(280, unit=u.nm), None, None, None, None])

A crop takes one entry per world coordinate of the cube: wavelength, sky
position, time and raster step, and raster scan for the cube of a whole window,
which has a raster axis. ``None`` leaves one uncropped. The sky position, time
and raster step (and scan) must be given together.

Building a cube
===============

Only the data and the WCS of `~irispy.spectrograph.SpectrogramCube` can be given
positionally, as for `~irispy.sji.SJICube`:

.. code-block:: python

    # Old
    SpectrogramCube(data, wcs, uncertainty, unit, meta)

    # New
    SpectrogramCube(data, wcs, uncertainty=uncertainty, unit=unit, meta=meta)

Memmap reads
============

Cubes read with ``memmap=True`` hold the unscaled integers of the file, in
``DN_IRIS_FUV_UNSCALED`` or ``DN_IRIS_NUV_UNSCALED``, with the window's ``BSCALE`` and ``BZERO``
in ``meta``. Their fill mask is a lazy Dask array, computed only for the slices that are used.

Several raster files read with ``memmap=True`` give one dask-backed cube that
reads each file on demand.
