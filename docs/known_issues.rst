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

Raster gWCS Inverse and Time
==============================

The raster gWCS exposes five world coordinates: wavelength, helioprojective longitude, helioprojective latitude, elapsed time, and an explicit scan-step index.
Because time may not be monotonic across flipped or repeated rasters, the WCS inverse (``world_to_array_index``) deliberately ignores the time component and resolves pixels using sky position plus the explicit scan-step coordinate only.
Consequently, passing a world tuple with a mismatched time value to ``world_to_array_index`` or ``crop`` will silently return the same pixel as if the correct time had been supplied.
To avoid confusion, prefer building crop bounds with ``cube.wcs.array_index_to_world`` rather than hand-constructing world tuples with arbitrary time values.
