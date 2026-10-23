.. _irspy_known_issues:

************
Known Issues
************

This page documents commonly known issues.
"Issues" here is defined broadly and refers to oddities or specifics of how ``irispy`` or the Python ecosystem works that can and will catch users off guard.

Per-exposure WCS metadata
=========================

The IRIS FITS files contain the per-exposure WCS metadata (reference coordinate and PCij matrix) in an extension, while the primary header has only values averaged over the observation.
For example, OBS 4204700138 SJI file, ``CRVAL`` in the primary header is (1.93840, 1.96290), but the reference coordinate
is actually (-3.56638378, 1.69258388), which is retrieved from the appropriate index in the ``XCENIX``/``YCENIX`` arrays in the extension.

The slit-jaw and spectrograph readers build the WCS of each cube from the values of every exposure, so use the cube's WCS rather than the primary header for the pointing.

gWCS observation time
=====================

The world coordinates from `irispy.sji.SJICube.wcs` (a gWCS) are in a `~sunpy.coordinates.Helioprojective` frame whose ``obstime`` is the time of the first exposure, whichever exposure they belong to.
The coordinates themselves use the pointing of their own exposure, and the time returned with them is the time of their own exposure; only the frame's ``obstime``, and so the observer it implies, is fixed.
This matters when transforming to frames that depend on the time, such as heliographic coordinates.
A gWCS output frame cannot yet change its ``obstime`` from one pixel to the next (`#43 <https://github.com/LM-SAL/irispy/issues/43>`__).
The frame of a raster cube's gWCS is likewise at the ``DATE_OBS`` of its file, or of the first file for several raster files read together.

If you need a coordinate frame at the time of a given exposure, use the FITS WCS of that exposure, ``sji_cube.fits_wcs[i]`` (see `irispy.sji.SJICube.fits_wcs`), or its map, ``sji_cube.to_maps(i)`` (see `irispy.sji.SJICube.to_maps`); both use the time of that exposure.

Memory use when reading data
============================

Level 2 files store 16-bit integers with a scale and offset (``BSCALE`` and ``BZERO``).
`irispy.io.read_files` loads the scaled values into memory as 32-bit floats, so the data take twice the size of the file and the mask of fill values another half of it.
For example, reading every spectral window of a 630 MB raster file holds about 1.6 GB of memory, and briefly about 1.9 GB while reading.

To use less memory:

* Read only the spectral windows you need with ``spectral_windows``; the memory goes down in proportion.
* Leave ``uncertainty=False`` (the default); ``uncertainty=True`` adds an array of the data's type and size.
* Pass ``memmap=True``, which maps the raw, unscaled integers instead of allocating scaled data numbers (DN); AIA and Hinode/SOT data are stored unscaled and keep their units.
  A lazy Dask mask marks both fill values without reading the mapped data at open.
  Only the mask slices that are used read data; computing the entire mask takes one byte per pixel.
  No uncertainty is computed in this mode.
  Most analysis functions refuse these raw IRIS data and need ``memmap=False``, the only mode that marks saturated samples (see :ref:`irispy-tutorial-saturated-data`).

Compressed SJI files are decompressed into memory once, so ``memmap=True`` cannot provide disk-backed arrays for them.

Loading the scaled and masked data lazily, only when they are used, is tracked in `#14 <https://github.com/LM-SAL/irispy/issues/14>`__.

.. _irispy-known-issues-extracted-archives:

Disk use of extracted archives
==============================

`irispy.io.read_files` extracts the raster, SDO and Hinode/SOT tarballs to disk before reading them, each into its own folder in the ``irispy`` cache folder, ``~/.cache/irispy`` unless ``XDG_CACHE_HOME`` is set.
The extracted files take about twice the space of the archive, and up to three times: a 913 MB raster tarball extracts to 2.0 GB.
They stay there until you delete them; an archive that is copied, moved or renamed is extracted again into a new folder, and the old folder stays.

To use less disk space:

* Delete the folders of the archives you no longer read, or the whole cache folder.
* Pass ``extract_dir`` to extract the archives somewhere else, such as a scratch disk.
  With ``memmap=False`` (the default) the data are read into memory, so ``extract_dir`` can be a `tempfile.TemporaryDirectory` on a disk with room (``dir=``), deleted with the extracted files at the end of its ``with`` block.

Searching the IRIS archive
==========================

Searches of the IRIS archive run in parallel, such as from several threads or processes, can return wrong start and end times; run one at a time.
