***********************************************
Supported API (`irispy` supported interfaces)
***********************************************

One canonical import path exists per supported operation. The tables below are the
supported surface; anything else in the API reference is implementation detail and
may change without notice. Every transformation returns a new object and leaves its
input unchanged, and symmetric errors travel as typed uncertainty objects
(`~astropy.nddata.StdDevUncertainty`, `~astropy.nddata.VarianceUncertainty`, or
`~astropy.nddata.InverseVariance`); unknown uncertainty types raise `TypeError`.

Readers and containers
======================

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Import
     - Input
     - Output
     - Uncertainty
   * - ``irispy.io.read_files``
     - Level 2 file path(s), one observation per call
     - Flat `~ndcube.NDCollection` of cubes and window sequences, keyed by product name
     - Typed, with ``uncertainty=True``; rejected with ``raw=True``
   * - ``irispy.io.read_sji_lvl2``
     - SJI level 2 file
     - `~irispy.sji.SJICube`
     - Typed, with ``uncertainty=True``
   * - ``irispy.io.read_spectrograph_lvl2``
     - Raster level 2 file
     - `~irispy.spectrograph.RasterCollection` of window sequences
     - Typed, with ``uncertainty=True``
   * - ``irispy.io.read_mosaic``
     - Mosaic level 2 file
     - `~irispy.sji.SJICube`
     - Typed, with ``uncertainty=True``
   * - `~irispy.sji.SJICube`, `~irispy.spectrograph.SpectrogramCube`
     - ``data, wcs`` then keyword-only options
     - The cube itself
     - Typed; `~astropy.nddata.UnknownUncertainty` raises `TypeError`

Calibration and analysis
========================

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Import
     - Input
     - Output
     - Uncertainty
   * - `~irispy.utils.response.get_response`
     - Scalar or array `~astropy.time.Time`
     - `dict`; ``DATE_OBS``, ``AREA_SG``, ``AREA_SJI`` carry a leading time axis for array input
     - Quantities carry their units
   * - `~irispy.utils.spectrograph.radiometric_calibration`
     - Scaled `~irispy.spectrograph.SpectrogramCube`
     - New radiance cube with ``response_version`` in its metadata
     - Typed in, `~astropy.nddata.StdDevUncertainty` out
   * - `~irispy.utils.moments.calculate_moments`
     - Scaled `~irispy.spectrograph.SpectrogramCube`
     - `~ndcube.NDCollection` of `~ndcube.NDCube` maps
     - Typed in, `~astropy.nddata.StdDevUncertainty` out
   * - `~irispy.utils.red_blue.calculate_red_blue_asymmetry`
     - Scaled `~irispy.spectrograph.SpectrogramCube`
     - `~ndcube.NDCollection` of maps and profile cubes
     - Asymmetry map carries `~astropy.nddata.StdDevUncertainty`
   * - `~irispy.utils.fitting.maps_from_fit`
     - Fitted model and its cube
     - `~ndcube.NDCollection` of maps plus a spectral residual
     - Covariance propagated to `~astropy.nddata.StdDevUncertainty`
   * - `~irispy.utils.density.density_diagnostic`
     - Line intensities, ``density_grid``, and a ``fiasco`` ion
     - `dict` with ``density`` and asymmetric ``density_lower``/``density_upper``
     - Asymmetric bounds preserved
   * - `~irispy.utils.dust.mask_dust`, `~irispy.utils.dust.remove_dust`
     - `~irispy.sji.SJICube`, 2D or 3D
     - New masked or repaired cube; ``meta["dust_masked"]`` records masking
     - Copied with the new cube

Removed and renamed interfaces
==============================

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Removed
     - Replacement
   * - ``irispy.utils.response.get_latest_response``
     - `~irispy.utils.response.get_response` with a required ``observation_time``
   * - ``SJICube.apply_dust_mask`` and its undo
     - `~irispy.sji.SJICube.mask_dust` and `~irispy.sji.SJICube.remove_dust`; keep the original cube to reverse
   * - ``dust_masked`` attribute
     - ``meta["dust_masked"]``
   * - ``red_blue_asymmetry_error`` result key
     - The ``.uncertainty`` of the ``"red_blue_asymmetry"`` map
   * - Moment keys ``intensity``, ``width``, ``velocity_width``
     - ``summed_intensity`` (or ``integrated_intensity``), ``sigma``, ``sigma_velocity``
   * - ``memmap=True`` returning raw values
     - ``raw=True``; ``memmap`` now only selects storage
   * - Positional ``SpectrogramCube`` options after ``data, wcs``
     - Keyword-only arguments, as on `~irispy.sji.SJICube`
