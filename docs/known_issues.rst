.. _irspy_known_issues:

************
Known Issues
************

This page documents commonly known issues.
"Issues" here is defined broadly and refers to oddities or specifics of how ``irispy`` or the Python ecosystem works that can and will catch users off guard.

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

Updating Raster ``crop`` Calls
==============================

IRIS spectrograph rasters now use a full gWCS, so raster crops must be written against the world-object order returned by ``cube.wcs.array_index_to_world(step, slit, spectral)``.
For rasters that order is ``(spectral, skycoord, time, scan_step)``.

The old two-component shorthand is no longer supported:

.. code-block:: python

   cube.crop([SpectralCoord(140.277 * u.nm), None], [SpectralCoord(140.277 * u.nm), None])
   cube.crop([None, target_skycoord], [None, target_skycoord])

Use one of these patterns instead.

Crop a spectroheliogram at one wavelength
-----------------------------------------

If you only want to constrain wavelength, pass the full four-component raster world tuple and use ``None`` for the unconstrained axes:

.. code-block:: python

   line_core = SpectralCoord(140.277 * u.nm)
   image = cube.crop(
       [line_core, None, None, None],
       [line_core, None, None, None],
   )


Crop a subcube from known raster corners
----------------------------------------

If you already know the pixel corners you want, convert them to full world tuples and pass those directly into ``crop``:

.. code-block:: python

   start = cube.wcs.array_index_to_world(50, 200, 10)
   stop = cube.wcs.array_index_to_world(150, 600, 25)
   subcube = cube.crop(start, stop)

Extract a spectrum at a sky position
------------------------------------

If you start from a ``SkyCoord``, first locate the nearest raster pixel on the sky grid, then convert that pixel back into the full world tuples expected by ``crop``:

.. code-block:: python

   target = SkyCoord(-338 * u.arcsec, 275 * u.arcsec, frame=target_frame)
   lon = cube.axis_world_coords_values("custom:pos.helioprojective.lon")[0].to_value(u.arcsec)
   lat = cube.axis_world_coords_values("custom:pos.helioprojective.lat")[0].to_value(u.arcsec)
   distance = (lon - target.Tx.to_value(u.arcsec)) ** 2 + (lat - target.Ty.to_value(u.arcsec)) ** 2
   step_index, slit_index = np.unravel_index(np.nanargmin(distance), distance.shape)

   start = cube.wcs.array_index_to_world(step_index, slit_index, 0)
   stop = cube.wcs.array_index_to_world(step_index, slit_index, cube.data.shape[-1] - 1)
   spectrum = cube.crop(start, stop)
