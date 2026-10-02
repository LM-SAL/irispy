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

This was (and partially still is) a problem for the ``irispy`` package, which assumed that the reference coordinate is the same for all exposures.
This has been addressed by the SJI reader but not fully by the spectrograph reader.

In the future, the goal is to create a gWCS to account for this.

Spectrogram WCS
===============

For different programs that are run for the slit spectrograph, a sit-and-stare has the CDELT of 0 arcseconds in the X direction.
This is not allowed by the WCS standard, so we set its value to 1e-10 arcsec in the X direction, which essentially tricks the WCS calculation to get it to work if there is no rotation.
However, the PCij matrix used is derived from the SJI, with square pixels, so the PCij matrix is a pure rotation.
This means that one gets the correct answer only if one does the matrix multiplication in the wrong order: first by PCij and then by CDELTs.

We work around this by modifying the PC_ij matrix to have the correct skew.
Since the X CDELT is 1e-10 arcsec, the inverse is thankfully not infinity.
Using equation 187 in `Calabretta & Greisen 2002 <https://www.aanda.org/articles/aa/abs/2002/45/aah3860/aah3860.html>`__, we correct for this.

Note that since these pixels are extremely rectangular, with an aspect ratio of ~3e-10, the cross terms in the PCij matrix are quite small: -3.4e-12 and -3.8e-7.
Hopefully, 64-bit floats have enough precision to enable this to work all of the time.

Slit-jaw gWCS observation time
==============================

The world coordinates from `irispy.sji.SJICube.wcs` (a gWCS) are in a `~sunpy.coordinates.Helioprojective` frame whose ``obstime`` is the time of the first exposure, whichever exposure they belong to.
The coordinates themselves use the pointing of their own exposure, and the time returned with them is the time of their own exposure; only the frame's ``obstime``, and so the observer it implies, is fixed.
This matters when transforming to frames that depend on the time, such as heliographic coordinates.
A gWCS output frame cannot yet change its ``obstime`` from one pixel to the next (`#43 <https://github.com/LM-SAL/irispy/issues/43>`__).

If you need a coordinate frame at the time of a given exposure, use the FITS WCS of that exposure, ``sji_cube.fits_wcs[i]`` (see `irispy.sji.SJICube.fits_wcs`), or its map, ``sji_cube.to_maps(i)`` (see `irispy.sji.SJICube.to_maps`); both use the time of that exposure.

Memory use when reading data
============================

Level 2 files store 16-bit integers with a scale and offset (``BSCALE`` and ``BZERO``).
`irispy.io.read_files` loads the scaled values into memory as 32-bit floats, so the data take twice the size of the file and the mask of fill values another half of it.
For example, reading every spectral window of a 630 MB raster file holds about 1.6 GB of memory, and briefly about 1.9 GB while reading.

To use less memory:

* Read only the spectral windows you need with ``spectral_windows``; the memory goes down in proportion.
* Leave ``uncertainty=False`` (the default); ``uncertainty=True`` adds a 64-bit array, twice the size of the data.
* Pass ``memmap=True``, which maps the raw, unscaled integers instead of allocating scaled data numbers (DN).
  A mask marks both fill values and takes one byte per pixel; creating it reads the data without modifying them.
  No uncertainty is computed in this mode.

Compressed SJI files are decompressed into memory once, so ``memmap=True`` cannot provide disk-backed arrays for them.

Loading the scaled and masked data lazily, only when they are used, is tracked in `#14 <https://github.com/LM-SAL/irispy/issues/14>`__.
