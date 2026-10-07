0.9.1 (2026-09-27)
==================

Breaking Changes
----------------

- The minimum supported versions are now astropy 8.0.0 and mpl-animators 1.2.3. (`#184 <https://github.com/LM-SAL/irispy/pull/184>`__)
- Removed ``SJICube.basic_wcs``; use ``SJICube.fits_wcs``, which now also describes the sliced pixels of a spatially sliced cube, and is ``None`` for a rebinned one. Removed ``SJICube.scaled`` and the ``scaled`` argument of `~irispy.sji.SJICube`; `~irispy.io.sji.read_sji_lvl2` stores it in ``cube.meta["scaled"]``. (`#186 <https://github.com/LM-SAL/irispy/pull/186>`__)
- Slicing a spectrograph or SJI cube so that an axis has length 0, for example ``cube[3:3]``, now raises ``IndexError`` instead of returning an empty cube. (`#186 <https://github.com/LM-SAL/irispy/pull/186>`__)


New Features
------------

- Added ``SGMeta.observer``, the IRIS observer (at Earth) at the start of the observation. (`#185 <https://github.com/LM-SAL/irispy/pull/185>`__)
- Added `~irispy.sji.SJICube.celestial_frame`, the `~sunpy.coordinates.frames.Helioprojective` frame of the observation. (`#186 <https://github.com/LM-SAL/irispy/pull/186>`__)
- `~irispy.io.read_files` reuses a complete earlier extraction of a tar file instead of extracting it again on every call, reading slit-jaw and AIA files builds their per-frame WCS headers much faster, and `~irispy.utils.dust.remove_dust` computes its spatial fallback only at the pixels it fills. (`#188 <https://github.com/LM-SAL/irispy/pull/188>`__)
- `~irispy.spectrograph.SpectrogramCube` no longer requires ``uncertainty``, ``unit`` and ``meta``; like `~ndcube.NDCube`, they default to None, so a cube can be made from just data and a WCS. (`#189 <https://github.com/LM-SAL/irispy/pull/189>`__)


Bug Fixes
---------

- Reading a V34 raster (``STEPS_AV < -0.01``) now flips its mask, uncertainty and per-step metadata (auxiliary times, exposure time, exposure FOV center, observer radial velocity and orbital phase) along with the data. Before, the mask and uncertainty came from the unflipped data, and the metadata stayed in file order. (`#176 <https://github.com/LM-SAL/irispy/pull/176>`__)
- Raster WCS lookups no longer slow down along the step axis: the -TAB table no longer has an explicit step index, which wcslib searched linearly, so a pixel-to-world call now costs the same at every step. (`#177 <https://github.com/LM-SAL/irispy/pull/177>`__)
- SJI and AIA cube coordinates are no longer one pixel off the FITS header: the gWCS now places the reference pointing at the FITS reference pixel ``CRPIX - 1`` in 0-based pixels. The ``slit x position`` and ``slit y position`` extra coordinates are now in pixels (their physical type becomes ``custom:PIXEL``) and ``ophaseix`` is dimensionless; all three were tagged arcsec. The slit positions keep the file's 1-based FITS values (``SLTPX1IX``/``SLTPX2IX``), so subtract 1 for 0-based pixel coordinates, as the SJI-SG slit example now does. (`#178 <https://github.com/LM-SAL/irispy/pull/178>`__)
- ``calculate_uncertainty``, used when reading with ``uncertainty=True``, now gives negative counts the readout noise instead of NaN and an "invalid value encountered in sqrt" warning. (`#179 <https://github.com/LM-SAL/irispy/pull/179>`__)
- Metadata properties no longer raise ``TypeError`` when the header lacks their keyword; they return `None`, like the other header properties. This fixes ``observer_radial_velocity`` on real Level 2 files (which have no ``OBS_VR``; rasters now return the per-exposure values), and ``spectral_range``, ``temporal_cadence`` and ``str(meta)`` on the IRIS-aligned AIA cutouts. ``observing_mode_id`` now reads the OBSID from the AIA cutouts' ``DATE_TIME_OBSID`` instead of turning it into one large number. (`#180 <https://github.com/LM-SAL/irispy/pull/180>`__)
- Reading a raster with ``memmap=True`` no longer computes an uncertainty from the unscaled data, as documented; before, every sample got the readout noise alone. (`#181 <https://github.com/LM-SAL/irispy/pull/181>`__)
- Negative indices now slice spectrograph and SJI cubes correctly: ``cube[-1]`` had NaN coordinates, ``cube[1:5][-1]`` described ``cube[0]``, and ``SJICube.to_maps(-1)`` gave the map an invalid ``DATE-OBS`` (-4713-11-24). (`#186 <https://github.com/LM-SAL/irispy/pull/186>`__)
- The outer pixel corners of SJI cubes now have finite times. (`#186 <https://github.com/LM-SAL/irispy/pull/186>`__)
- Printing a `~irispy.spectrograph.RasterCollection` now lists its aligned physical types in a fixed order; it changed from run to run before. (`#188 <https://github.com/LM-SAL/irispy/pull/188>`__)
- `~irispy.sji.SJICube.to_maps` now sets the wavelength of the maps, so their names and plot titles show it. (`#190 <https://github.com/LM-SAL/irispy/pull/190>`__)
- Uncertainties computed with ``uncertainty=True`` (and by `~irispy.utils.spectrograph.radiometric_calibration`) are now stored as `~astropy.nddata.StdDevUncertainty` instead of an unknown uncertainty type. (`#191 <https://github.com/LM-SAL/irispy/pull/191>`__)


Documentation
-------------

- The gallery examples now download cut-down copies of their IRIS rasters, holding only the scans and spectral windows each example uses, from `LM-SAL/irispy-data <https://github.com/LM-SAL/irispy-data>`__. Each example links the full observation in the IRIS archive. (`#187 <https://github.com/LM-SAL/irispy/pull/187>`__)
- The fitting examples average 2x2 spatial pixels before fitting, the reprojection example downloads its AIA image from `LM-SAL/irispy-data <https://github.com/LM-SAL/irispy-data>`__ instead of the VSO, and the AIA cube example uses a five-minute cutout of its observation. (`#189 <https://github.com/LM-SAL/irispy/pull/189>`__)
- Fixed the fitted and moment maps in the spectral fitting, spectral moments and Mg II two-Gaussian gallery examples, which were plotted against the wrong coordinates, the prose of the IRIS–AIA co-alignment example, which rendered as code, and the rolled-SJI reprojection example, which matched the AIA image to a different SJI frame. (`#189 <https://github.com/LM-SAL/irispy/pull/189>`__)
- Fixed wrong results in several gallery examples: the slit-jaw light curve and date axis in the umbral flashes example, the Mg II k core wavelength and v34 flip description in the raster how-to examples, the red-blue asymmetry windows and sign, the color map of the double-Gaussian asymmetry map, the axis labels and color scale of the Mg II Dopplergram, the AIA time used for co-alignment, and several units, labels and descriptions. (`#190 <https://github.com/LM-SAL/irispy/pull/190>`__)
- Added a "Citing irispy" section to the README and documentation, pointing at the Zenodo record of every release. (`#192 <https://github.com/LM-SAL/irispy/pull/192>`__)
- The API documentation no longer includes inheritance diagrams, so building the documentation no longer needs graphviz. (`#193 <https://github.com/LM-SAL/irispy/pull/193>`__)
- Documented that the frame of `irispy.sji.SJICube.wcs` uses the time of the first exposure as its ``obstime``, and how to get a frame for the time of each exposure. (`#194 <https://github.com/LM-SAL/irispy/pull/194>`__)
- Documented how much memory reading data uses and how to use less. (`#195 <https://github.com/LM-SAL/irispy/pull/195>`__)


Internal Changes
----------------

- The documentation build and the remote-data tests now download their data from release assets on `LM-SAL/irispy-data <https://github.com/LM-SAL/irispy-data>`__ instead of Git LFS, and the documentation uses an O IV-only CHIANTI database (35 MB instead of 909 MB). (`#183 <https://github.com/LM-SAL/irispy/pull/183>`__)


0.9.0 (2026-09-10)
==================

New Features
------------

- Added `irispy.utils.rgb.calculate_rgb` and `irispy.utils.rgb.plot_rgb`, also reachable as
  ``cube.plotter.plot_rgb()``, which render a `~irispy.spectrograph.SpectrogramCube` as a false-color
  image whose brightness and hue depend on intensity and spectral shape, so Doppler shifts show up as color.
  Wavelengths are converted to Doppler velocities around the rest wavelength, which defaults to the
  metadata ``TWAVE`` (an explicit rest wavelength is required when it is missing), and mapped to color through ``velocity_norm``, by default the Doppler velocity itself; ``asinh_velocity`` offers an arcsinh scale of 25 km/s to bring out small shifts.
  The mapped range defaults to +/-100 km/s when the rest wavelength lies within the cube's range,
  otherwise to the full window; explicit wavelength bounds override it.
  The image can be drawn against helioprojective longitude or against time, and the colorbar is
  labeled in both wavelength and Doppler velocity. The colorbar represents single-bin spectra;
  finite-width lines can have different colors at the same peak intensity.
  This needs the optional ``colorsynth`` dependency, installable with ``pip install 'irispy-lmsal[rgb]'``. (`#168 <https://github.com/LM-SAL/irispy/pull/168>`__)


Bug Fixes
---------

- Reading an SJI file no longer crashes when a pointing column is entirely zero
  (e.g. the PC off-diagonals of an unrotated observation), and valid zeros are
  no longer interpolated over. Only fully-zeroed pointing rows (dropped
  exposures) are treated as gaps, they are now handled consistently for both
  the gWCS and the header-based WCS, and are filled with the average of the
  neighboring exposures. (`#169 <https://github.com/LM-SAL/irispy/pull/169>`__)
- Fixed a crash when plotting a 1D slice of an SJI cube: the default colormap is no longer passed to the 1D line plot. (`#170 <https://github.com/LM-SAL/irispy/pull/170>`__)


Internal Changes
----------------

- The test-data generator ``compress.py`` was reworked: it now writes
  ``*_test.fits`` copies instead of overwriting its inputs, decimates by
  strided selection so bad-pixel and saturation sentinels survive exactly,
  keeps world coordinate spans and the exposure bookkeeping (auxiliary table,
  source-filename table, ``NEXP``) consistent, and the bundled test files were
  renamed to the ``_test`` convention and repaired accordingly. (`#169 <https://github.com/LM-SAL/irispy/pull/169>`__)
- IRIS plotters are now registered through ndcube's plotter framework, so ``cube.plotter`` is an IRIS plotter and extra plotter methods are reachable as ``cube.plotter.<method>()``; ``cube.plot()`` behaves as before. (`#170 <https://github.com/LM-SAL/irispy/pull/170>`__)
- Repaired the WCS of the bundled test FITS files. (`#171 <https://github.com/LM-SAL/irispy/pull/171>`__)


0.8.1 (2026-08-14)
==================

Bug Fixes
---------

- `~irispy.utils.spectrograph.radiometric_calibration` no longer returns infinite or
  wildly amplified values for wavelengths outside the spectral ranges covered by the
  response file (e.g., the blue end of a full-CCD Si IV window). (`#164 <https://github.com/LM-SAL/irispy/pull/164>`__)
- Fixed `~irispy.io.spectrograph.read_spectrograph_lvl2` (used by `~irispy.io.read_files`)
  assigning data to the wrong spectral window when ``spectral_windows`` was passed
  in an order different from the order of the windows in the file. (`#164 <https://github.com/LM-SAL/irispy/pull/164>`__)


Documentation
-------------

- The example gallery now executes examples in parallel (inheriting sphinx's ``-j``),
  significantly reducing documentation build times. Setting ``IRISPY_GALLERY_PROFILE=1``
  falls back to serial execution since memory profiling requires it. (`#164 <https://github.com/LM-SAL/irispy/pull/164>`__)


0.8.0 (2026-08-10)
==================

Breaking Changes
----------------

- Simplified `irispy.utils.red_blue.calculate_red_blue_asymmetry` by removing low-level tuning arguments that are now handled internally.
  The function now always centers profiles on the line peak, uses uncertainty from the input cube when available, and derives its interpolation window from ``velocity_range``.
  The function now takes ``degree`` instead of ``interpolation_kind``, and removes ``mask_negative`` (now always applied), ``uncertainty``, ``center_on_peak``, ``velocity_window``, and ``fit_window``.
  The output now contains only the red-blue asymmetry map, quality map, optional uncertainty map, and optional profile cubes. (`#144 <https://github.com/LM-SAL/irispy/pull/144>`__)


New Features
------------

- Add ``spectral_dispersion`` and ``solid_angle`` properties to ``SpectrogramCube``. (`#141 <https://github.com/LM-SAL/irispy/pull/141>`__)
- Exposed much more FITS metadata to the user via the ``meta``. (`#142 <https://github.com/LM-SAL/irispy/pull/142>`__)


Bug Fixes
---------

- Preserve the Level 2 spectrograph slit WCS instead of applying detector pointing offsets a second time, use detector-specific source-file timestamps for raster exposures, and expose the AUX-derived times in the cube metadata. (`#158 <https://github.com/LM-SAL/irispy/pull/158>`__)
- Use the per-step AUX pointing values in a FITS-TAB WCS for spectrograph rasters and sit-and-stare observations, and
  correct the observer coordinate and observation time stored in the WCS. (`#159 <https://github.com/LM-SAL/irispy/pull/159>`__)
- The time coordinates for SJI and spectrograph data are now the exposure midpoints (``T_OBS``, i.e., start time plus half the exposure time) instead of the exposure start times. (`#161 <https://github.com/LM-SAL/irispy/pull/161>`__)
- Spectrograph files with missing NUV exposures (``EXPTIMEN`` of 0 s) are now read instead of raising an error. (`#161 <https://github.com/LM-SAL/irispy/pull/161>`__)


Documentation
-------------

- Revamped both cosmic ray removal examples to split focus between SG and SJI data. (`#139 <https://github.com/LM-SAL/irispy/pull/139>`__)
- Correct and expand the SJI and spectrograph slit-coordinate comparison to show NUV and FUV detector offsets. (`#156 <https://github.com/LM-SAL/irispy/pull/156>`__)


0.7.0 (2026-05-07)
==================

New Features
------------

- Added `irispy.utils.dust.remove_dust` and `irispy.sji.SJICube.remove_dust` to repair dust-darkened pixels in IRIS slit-jaw images. (`#111 <https://github.com/LM-SAL/irispy/pull/111>`__)
- Added ``remove_cosmic_rays`` methods both for SJI and raster cubes, with two possible backends: ``rsliding`` and ``astroscrappy``. (`#113 <https://github.com/LM-SAL/irispy/pull/113>`__)
- Added a way to calculate the moments for a SpectrumCube with an example: :ref:`sphx_glr_generated_gallery_analysis_04_spectral_moments.py`. (`#124 <https://github.com/LM-SAL/irispy/pull/124>`__)
- Added utilities and examples to calculate moments and red-blue asymmetry maps for `~irispy.spectrograph.SpectrogramCube`. (`#125 <https://github.com/LM-SAL/irispy/pull/125>`__)
- Added utilities to compute IRIS density diagnostics from line ratios, including `~irispy.utils.density.density_diagnostic` and `~irispy.utils.density.map_ratio_to_quantity`. (`#126 <https://github.com/LM-SAL/irispy/pull/126>`__)


Bug Fixes
---------

- Fixed a few places with incorrect timestamp handling for the gWCS. (`#105 <https://github.com/LM-SAL/irispy/pull/105>`__)
- Fixed `irispy.sji.SJICube` slicing so ``basic_wcs`` is preserved for common slicing operations, allowing sliced cubes to continue working with `irispy.sji.SJICube.to_maps` and other WCS-based workflows. (`#111 <https://github.com/LM-SAL/irispy/pull/111>`__, `#112 <https://github.com/LM-SAL/irispy/pull/112>`__)
- Fixed `irispy.utils.spectrograph.radiometric_calibration` so sliced raster `irispy.spectrograph.SpectrogramCube` objects continue to calibrate correctly when slicing changes the WCS wrapper or reduces the data to two dimensions. (`#114 <https://github.com/LM-SAL/irispy/pull/114>`__)
- Removed ``preserve_units=True`` WCS construction paths due to an upstream bug in Astropy. (`#117 <https://github.com/LM-SAL/irispy/pull/117>`__)
- Fixed raster file-list reading so plain spectrograph FITS lists load the full observation sequence. (`#117 <https://github.com/LM-SAL/irispy/pull/117>`__)
- Fixed several edge cases in SJI and spectrograph coordinate preservation, bad-pixel masking, density diagnostics, spectral moments, dust masking, and red-blue asymmetry memory usage. (`#129 <https://github.com/LM-SAL/irispy/pull/129>`__)


Documentation
-------------

- Added a tutorial mirroring how ITN26 is written. (`#106 <https://github.com/LM-SAL/irispy/pull/106>`__)
- Added an example of using `astroscrappy` to remove cosmic rays from IRIS data: :ref:`sphx_glr_generated_gallery_calibration_01_remove_spikes_sg.py`. (`#106 <https://github.com/LM-SAL/irispy/pull/106>`__)
- Added a how-to example showing how to remove dust from IRIS slit-jaw images with `irispy.sji.SJICube.remove_dust`: :ref:`sphx_glr_generated_gallery_calibration_04_remove_dust.py`. (`#111 <https://github.com/LM-SAL/irispy/pull/111>`__)
- Added a double Gaussian Mg II k spectral fitting example: :ref:`sphx_glr_generated_gallery_analysis_07_mg_ii_two_gaussian_fitting.py`. (`#130 <https://github.com/LM-SAL/irispy/pull/130>`__)


0.6.0 (2026-01-26)
==================

Breaking Changes
----------------

- Moved internal code to move to use "DATE_OBS" instead of "STARTDATE". (`#96 <https://github.com/LM-SAL/irispy/pull/96>`__)
- Removed support for Python 3.11. (`#97 <https://github.com/LM-SAL/irispy/pull/97>`__)
- Made the standard WCS lazy created on access now to speed up loading (`#103 <https://github.com/LM-SAL/irispy/pull/103>`__)


Documentation
-------------

- Added an example to demonstrate co-alignment of IRIS SJI and SDO/AIA images using sunkit-image's ``match_template`` method. (`#65 <https://github.com/LM-SAL/irispy/pull/65>`__)
- Added an offset example, attempting to track down a WCS offset between the IRIS SJI and IRIS SG WCS. (`#96 <https://github.com/LM-SAL/irispy/pull/96>`__)


0.5.0 (2025-10-06)
==================

Breaking Changes
----------------

- `irispy.utils.spectrograph.radiometric_calibration` function was added, replacing ``convert_between_dn_and_photons``. (`#77 <https://github.com/LM-SAL/irispy/pull/77>`__)
- ``fitsinfo`` now renamed to `irispy.io.fits_info`.
  The output has been updated to be nicer. (`#78 <https://github.com/LM-SAL/irispy/pull/78>`__)
- Renamed "wobble_movie" to "generate_wobble_movie". (`#88 <https://github.com/LM-SAL/irispy/pull/88>`__)


Bug Fixes
---------

- Improved radiometric calibration calculations and fixed unit conversion errors.
  It still does not match the IDL code, it can over estimate the radiance by a small margin. (`#77 <https://github.com/LM-SAL/irispy/pull/77>`__)


Internal Changes
----------------

- Retemplated package using the sunpy template. (`#88 <https://github.com/LM-SAL/irispy/pull/88>`__)


0.4.0 (2025-08-28)
==================

Breaking Changes
----------------

- Renamed "get_iris_response" to `irispy.utils.response.get_latest_response`. (`#74 <https://github.com/LM-SAL/irispy/pull/74>`__)
- Removed versions from the response, it now only supports the latest version which is currently at V9. (`#74 <https://github.com/LM-SAL/irispy/pull/74>`__)
- Renamed ``Collection`` to ``RasterCollection``.
  Moved metadata into a separate file.
  Removed ``convert_to`` method. (`#81 <https://github.com/LM-SAL/irispy/pull/81>`__)
- Increased minimum version of Python to 3.11, sunpy to 7.0.0 and dkist to 1.15.0 (`#81 <https://github.com/LM-SAL/irispy/pull/81>`__)
- All references to irispy-lmsal have been removed and now the package is simply referred to as ``irispy``.
  The package name on PyPI and conda-forge is still the same due to conflicts with existing packages. (`#85 <https://github.com/LM-SAL/irispy/pull/85>`__)


Internal Changes
----------------

- Added more test data. (`#86 <https://github.com/LM-SAL/irispy/pull/86>`__)


0.3.1 (2025-07-30)
==================

New Features
------------

- Added ``fits_header`` property to the ``.meta``. (`#69 <https://github.com/LM-SAL/irispy/pull/69>`__)
- Added observer location to the Spectrograph cubes.
  This now matches the SJI cubes. (`#70 <https://github.com/LM-SAL/irispy/pull/70>`__)


Bug Fixes
---------

- Fixed handling of raster tarfiles to not just use the first file. (`#69 <https://github.com/LM-SAL/irispy/pull/69>`__)
- Improved how OBSID v34 is checked by using ``STEPS_AV`` in the FITS header. (`#69 <https://github.com/LM-SAL/irispy/pull/69>`__)
- Improved how ``read_files`` handles tarfiles and mixed files. (`#69 <https://github.com/LM-SAL/irispy/pull/69>`__)


0.3.0 (2025-06-16)
==================

Breaking Changes
----------------

- Now ``read_files`` will return a NDCollection which you can access individual cubes based on keys like a dictionary.
  If one item was passed into ``read_files``, that return type has been unchanged. (`#63 <https://github.com/LM-SAL/irispy/pull/63>`__)
- Increased minimum version of dkist to 1.11.0. (`#63 <https://github.com/LM-SAL/irispy/pull/63>`__)
- Increased minimum version of sunraster to 0.6.0. (`#65 <https://github.com/LM-SAL/irispy/pull/65>`__)


New Features
------------

- Added explicit support for the IRIS aligned AIA cubes provided by LMSAL for each IRIS observation. (`#63 <https://github.com/LM-SAL/irispy/pull/63>`__)
- Added ``to_maps`` to ``SJICubes`` to allow a user to output a sunpy Map or MapSequence based on how many slices they need. (`#64 <https://github.com/LM-SAL/irispy/pull/64>`__)


0.2.5 (2025-06-02)
==================

Documentation
-------------

- Added raster v34 example

Internal Changes
----------------

- Updated slider names on plots

0.2.4 (2025-05-08)
==================

Documentation
-------------

- Simplified the spectral fitting example by making it single threaded.

0.2.3 (2025-05-07)
==================

Internal Changes
----------------

- Reduced sunpy minimum version to 6.0 from 6.1

0.2.2 (2025-05-07)
==================

Documentation
-------------

- Rewrote existing examples to be more consistent.
- Added an example for single Gaussian fitting using new functionality from astropy 7.0

Internal Changes
----------------

- Rewrite of unit tests.
- Fixed warning from DKIST modeling.

0.2.1 (2024-06-09)
==================

Internal Changes
----------------

- Add COC, add more links to docs and IO section

0.2.0 (2023-12-25)
==================

Features
--------

- Add support for V34 files.

Breaking Changes
----------------

- SJI data is now stored using a gWCS.
- All keywords have to passed by name into to all functions now.
- Dropped Python 3.8 support.

Internal Changes
----------------

- Templated to remove setup.py and setup.cfg
- Tweaks to documentation.

0.1.5 (2022-10-12)
==================

Bug Fixes
---------

- Fixed Windows path issue for wobble movie

0.1.4 (2022-09-26)
==================

Features
--------

- Added a timestamp to each frame of the wobble movie.
  You will need to set the ``timestamp`` keyword to be `True`.
- Added a ``wobble_cadence`` keyword to override the default wobble cadence of 180 seconds.

0.1.3 (2022-05-22)
==================

Features
--------

- Added V5 and V6 support for ``get_iris_response``. It also does not download the files anymore.

Breaking Changes
----------------

- API of ``get_iris_response`` has changed:
  ``pre_launch`` has gone, use ``response_version=2`` instead.
  ``response_file`` keyword has been removed, it will use files provided by the package instead.
  ``force_download`` was removed as the function now does not download any files.

0.1.2 (2022-05-02)
==================

Features
--------

- Tweaked ``irispy.utils.wobble_movie`` to remove limits on the metadata.
- Pin ``sunraster`` version due to Python version incompatibilities.

0.1.1 (2022-02-17)
==================

Features
--------

- Added a ``irispy.utils.wobble_movie`` to create a wobble movie. It does need FFMPEG to be installed.

0.1.0 (2022-01-14)
==================

First formal release of ``irispy``.

Please note there are parts of this library that are still under going development and will be updated as time
goes on.
There is also a lot of work to be done on the documentation and some of the functions in the ``utils`` module
do not function.
