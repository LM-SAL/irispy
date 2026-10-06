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
  A lazy Dask mask marks both fill values without reading the mapped data at open.
  Only the mask slices that are used read data; computing the entire mask takes one byte per pixel.
  No uncertainty is computed in this mode.

Compressed SJI files are decompressed into memory once, so ``memmap=True`` cannot provide disk-backed arrays for them.

Loading the scaled and masked data lazily, only when they are used, is tracked in `#14 <https://github.com/LM-SAL/irispy/issues/14>`__.

SJI burst detection and IDL
===========================

The Level 2 readers mask both ``-200`` and ``-199`` as missing data, including in AIA cutouts.
The ``-199`` convention comes from IRIS SolarSoft, including ``iris_make_fits_level3`` v1.29 and ``iris_raster_browser``.
The IDL SJI burst reference includes ``-199`` in its statistics, so excluding it can lower the detection threshold slightly.

Offsets between FUV and NUV windows along the slit
==================================================

The FUV and NUV spectra are recorded on different detectors, and the Level 2 pipeline shifts the FUV spectra along the slit so that their fiducial marks line up with those of the NUV spectra :cite:p:`wulser2018`.
The shift is recorded in the ``HISTORY`` of each file as "FUVS Fiducial midpoint Y shift" and "FUVL Fiducial midpoint Y shift".
We measured how well the windows line up afterwards in ten observations from 2013 to 2026, a small sample rather than a survey of the archive.
`irispy.utils.fiducials.find_fiducials` gives the row of each fiducial mark in each spectral window, and we compared those rows with the rows in Mg II k.
The pipeline had shifted the FUV spectra of these observations by 0.7 to 16.7 pixels.

In seven of the ten observations no FUV window shows a mark, so the residual offset cannot be measured from the spectra.
In the other three the FUV windows are within 0.2 pixels of Mg II k.
A pixel is 0.16635″ and a positive offset puts the mark at a higher row than in Mg II k:

.. list-table::
   :header-rows: 1

   * - Observation
     - Offset (pixels)
   * - 2014-07-08, OBS 3824262996
     - C II 1336 −0.08, O I 1356 +0.18, Si IV 1394 −0.10, Si IV 1403 +0.04
   * - 2015-01-30, OBS 3893010094
     - C II 1336 −0.17, Si IV 1394 −0.15
   * - 2021-04-29, OBS 3660259102
     - C II 1336 +0.03, Si IV 1403 +0.06

That is 0.03″ at most, a tenth of the spatial resolution of 0.33″ in the FUV and 0.4″ in the NUV :cite:p:`depontieu2014`.
The NUV windows of all ten observations are within 0.3 pixels of Mg II k.
If the alignment of the windows matters to your analysis, check your own observation the same way.
