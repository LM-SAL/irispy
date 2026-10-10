# irispy API refactor plan

Written 2026-10-09, assuming PR #235 (restore `+Inf` at the level 2 ceiling, drop the
saturation keywords) is merged. Upstream state on that date: ndcube 2.4.2, sunraster 0.7.0,
sunpy 8.0.1, astropy 8.0.1, dkist 1.18.1. ndcube PR 979 (negative-index slicing) is open.

## Principles decided

- **Functions over methods.** irispy's own operations are `function(cube, *args, **kwargs)`.
  Methods that come from ndcube and sunraster (`plot`, `crop`, `rebin`,
  `apply_exposure_time_correction`) stay until their package goes.
- **Metadata is the human-readable layer.** A property earns a name when a researcher would ask
  the question without knowing the FITS keyword and the answer is not in the cube or derivable
  from it. Everything else stays reachable as `meta["KEYWORD"]`.
- **One route per operation.** No method-and-function pairs, no two keywords for one effect.
- **Upstream first.** Use ndcube and sunpy features before writing irispy ones.
- **Breaking changes are acceptable**, one Towncrier `breaking` fragment per PR.
- **Leave alone**: the `calculate_` prefixes, the map key names, the dkist models (see item 10).

## Phase 1: cuts and renames (deletions, one small PR each)

### 1. Metadata properties: keep 44, cut 37

Keep, by the question they answer:

- What was observed: `observing_mode_id`, `observing_mode_description`, `observing_title`,
  `spectral_window`, `spectral_band`, `spectral_range`, `detector`, `detector_band`,
  `rest_wavelength`, `number_of_spectral_windows`.
- When: `date_start`, `date_end`, `date_reference`, `observing_campaign_start`,
  `observing_campaign_end`, `temporal_cadence`, `exposure_time`, `exposure_time_min`,
  `exposure_time_max`, `exposure_mu`, `automatic_exposure_control_enabled`.
- Where and how: `fov_center`, `raster_fov_width_x`, `raster_fov_width_y`, `mu`,
  `satellite_rotation`, `tracking_mode_enabled`, `number_of_raster_positions`,
  `number_of_unique_raster_positions`, `step_size_average`, `spatial_summing_factor`,
  `spectral_summing_factor`.
- Data quality: `observation_includes_saa`, `observatory_at_high_latitude`, `number_of_spikes`,
  `percent_data`.
- Provenance: `processing_level`, `reformat_version`.
- Required by sunraster or used internally: `instrument`, `observatory`, `distance_to_sun`,
  `observer_radial_velocity`, `sun_angular_radius`, `fits_header`.

Cut:

- Derivable from the cube: `data_mean`, `data_rms`, `data_median`, `data_min`, `data_max`,
  `window_mean`, `window_rms`, `window_median`, `window_min`, `window_max`, `data_unit`,
  `data_type`, `number_of_exposures`, `number_of_exposures_planned`,
  `number_of_exposures_observation`, `camera`.
- Always zero in level 2 files: `number_of_saturated_pixels`, `window_saturated_pixels`.
- Planner and pipeline internals: `observing_label`, `raster_repetition`, `raster_type_index`,
  `raster_type_total`, `lut_id`, `build_version`, `reformat_date`, `data_status`,
  `number_of_missing_raster_files`, `number_of_missing_observation_files`,
  `exposure_control_triggers_in_observation`, `exposure_control_triggers_in_raster`.
- Spread statistics where the average is kept: `step_size_stddev`, `step_time_average`,
  `step_time_stddev`, `cadence_planned_average`, `cadence_planned_stddev`,
  `cadence_executed_stddev`, `window_spikes`.

- [ ] Delete the properties in `irispy/meta.py` and their tests.
- [ ] Replace the `dir()` listing in `docs/tutorial/level_2.rst` with the grouped table above and
      one sentence on `meta["KEYWORD"]`.
- [ ] `breaking` fragment listing the removed names.

### 2. Helpers with no outside caller

- [ ] Delete `get_detector_type` (no caller; `irispy/utils/utils.py`).
- [ ] Make private and drop from `__all__` and the API reference: `record_to_dict`,
      `calculate_dust_mask`, `calculate_uncertainty`, `convert_photons_per_sec_to_radiance`,
      `calculate_dn_to_radiance_factor`, `reshape_1d_wavelength_dimensions_for_broadcast`,
      `map_ratio_to_quantity`, `find_bright_image_events`, `find_bright_spectral_events`.
- [ ] `image_clipping`: gallery pages use `astropy.visualization.AsymmetricPercentileInterval`;
      the helper goes private (still used by the wobble movie).
- [ ] Remove `_fit_xput_lite` from `irispy/utils/response.py` `__all__`.
- [ ] `generate_wobble_movie` stays public; add a gallery example (it has none).

### 3. Functions over methods

- [ ] Remove `SpectrogramCube.remove_cosmic_rays`, `SJICube.remove_cosmic_rays`,
      `SJICube.remove_dust`; the functions in `irispy.utils` are the one route.
- [ ] `SJICube.apply_dust_mask(undo=)` becomes `apply_dust_mask(cube, *, undo=False)` returning a
      new cube with the mask updated.
- [ ] `SJICube.to_maps(index)` and `MosaicCube.to_map(wavelength)` become one function
      `to_map(cube, ...)`: an integer or slice selects frames of an image cube (a `Map` for one
      frame, a `MapSequence` otherwise), a wavelength or range selects from a mosaic. Home decided
      in the PR (`irispy.utils.spectrograph` or a new `irispy.utils.maps`).
- [ ] Delete the `plot_rgb` wrapper on `SpectrogramPlotter`; `irispy.utils.rgb.plot_rgb(cube)`
      stays.
- [ ] Drop `integrated=` from `calculate_moments`; document
      `moments["intensity"] * cube.spectral_dispersion` and point to `average_window`.

### 4. Keyword vocabulary

- [ ] `calculate_red_blue_asymmetry(velocity_range=)` becomes `wing_range=(50, 150) km/s`, a
      positive band applied to both wings; `dv=` becomes `velocity_step=`.
- [ ] `find_si_iv_bursts` and the spectral event finder take `velocity_range=(-50, 50) km/s`, a
      signed pair as in `calculate_moments`, instead of a scalar half-width.
- [ ] After this, `velocity_range` means one thing everywhere: a signed `(lower, upper)` pair in
      km/s around `rest_wavelength`.

### 5. Exports and `fits_info`

- [ ] Add `read_spectrograph_lvl2` to `irispy.io.__all__`.
- [ ] `fits_info` returns its `astropy.table.Table` instead of printing: columns `No.`, `Name`,
      `Type`, `Dimensions`, `Description` (drop `Ver`, `Cards`, `Format`); `filename`,
      `observation` and `obsid` in `table.meta`. Tutorial pages add `print(...)`.

### 7. Adopt `NDCube.fill_masked`

- [ ] Replace the hand-rolled mask-to-NaN or zero blocks where it is a pure replacement:
      `bursts.find_bright_image_events`, `rgb.calculate_rgb`, `red_blue._prepare_data`,
      `wavelength_drift._line_shifts`. The fitting starts keep their own line (they also blank
      non-finite samples).
- [ ] Readers guarantee a boolean mask (`np.asarray(mask, dtype=bool)` at the boundary), since
      `fill_masked` indexes with the mask as given. Move the integer-mask guard test in
      `test_mg_features.py` to the readers.

### 13. `irispy.analysis` package

- [ ] Move the product-making modules: `moments`, `mg_features`, `red_blue`, `bursts`, `fitting`,
      `density`, `rgb`, `wavelength_drift`, `fiducials`. Re-export their entry points flat, so
      `from irispy.analysis import calculate_moments` works.
- [ ] `irispy.utils` keeps calibration and cleaning: `spectrograph` (radiometric calibration,
      background), `response`, `constants`, `cosmic_rays`, `dust`, `wobble`, and the private
      helpers.
- [ ] Split `docs/reference/utils.rst` into `analysis.rst` and `utils.rst`; update the gallery
      imports.

## Phase 2: structure

### 9b. Collapse the product classes (sunraster stays for now)

- [ ] Fold `AIACube` and `SOTCube` into `SJICube` and `MosaicCube` into `SpectrogramCube`. They
      differ by the word in `__str__`, one `BUNIT` branch for SOT, and `to_map`, which item 3
      makes a function. The product name comes from meta (`INSTRUME`, `LAMREF`).
- [ ] Remove their plotter registrations and the `isinstance(self, SOTCube)` branch in the
      slit-jaw map conversion.
- [ ] `breaking` fragment: `isinstance` checks on the three classes stop working.

### 12. One quality convention

- [ ] Shared leading values in every `_QualityFlag` enum: `OK = 0`, `NO_DATA = 1`,
      `SATURATED = 2`, `LOW_SIGNAL = 3`; analysis-specific flags follow.
- [ ] `calculate_moments` returns a `quality` map with those four, replacing the boolean
      `saturated` map and the NaN-only signal of `min_intensity`.
- [ ] `calculate_mg_features` returns `k_quality` and `h_quality` with `NOT_FOUND` added,
      replacing `{line}_saturated`.
- [ ] `RBAQualityFlag`, `FitQualityFlag` and `NonThermalQualityFlag` renumber their common values
      to match (`MASKED_INPUT`, `NO_FINITE_DATA` and `NO_DATA` are the same state).
- [ ] Value maps stay NaN wherever the flag is not `OK`; quality maps are never masked.

## Phase 3: additions

### 11. Fido client for the IRIS search

Backend facts, verified 2026-10-09: `https://www.lmsal.com/hek/hcr?cmd=search-events3&outputformat=json&instrument=IRIS&startTime=...&stopTime=...&hasData=true`
returns `{"Events": [...]}` with 113 fields per observation. The ones to use: `iris_obsid`,
`startTime`, `stopTime`, `goal`, `xCen`, `yCen`, `raster_fovx`, `raster_fovy`, `sji_fovx`,
`sji_fovy`, `parentUrl`. Raster and slit-jaw files sit at
`https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/YYYY/MM/DD/<date>_<time>_<obsid>/iris_l2_<date>_<time>_<obsid>_raster.tar.gz`
and `..._SJI_<band>_t000.fits.gz`.

- [ ] `irispy/net/`: a `sunpy.net.base_client.BaseClient` subclass registered with Fido.
- [ ] Attributes: `a.Time`, `a.Instrument("IRIS")`, `a.iris.ObsID`, `a.iris.Goal` (substring of
      the goal string); `a.Level` fixed at 2.
- [ ] `search` returns one row per observation: OBS ID, start, stop, goal, pointing, fields of
      view, and which products exist.
- [ ] `fetch` downloads the raster tarball and the slit-jaw bands requested (default: all),
      through sunpy's downloader; `read_files` already reads both.
- [ ] Tests with recorded responses, plus one `remote_data` smoke test. A tutorial page replaces
      the website walkthrough as the first route.

### 14. ASDF converters

ndcube's converters match exact classes (`NDCubeConverter.types == ["ndcube.ndcube.NDCube"]`),
so the irispy subclasses are not covered.

- [ ] Converters and tags for `SpectrogramCube`, `SJICube`, `SpectrogramCubeSequence`,
      `RasterCollection`, `SGMeta`, `SJIMeta`, `MosaicMeta`, registered through an
      `asdf.extensions` entry point. No save or load functions: `asdf.AsdfFile({...}).write_to`
      and `asdf.open` are the API, as for astropy and sunpy objects.
- [ ] Round-trip tests for a raster cube, a slit-jaw cube with its gwcs, and a moments collection.
- [ ] One tutorial section: saving processed products.

## Deferred, waiting on upstream

### 8. Delete the negative-index mixin

- [ ] When the ndcube release carrying PR 979 is out, raise the ndcube floor to it and delete
      `irispy/_wcs.py` (65 lines) and `_ResolveNegativeIndicesMixin` from both cube classes.

### 9a. When sunraster is retired

- [ ] Base `SpectrogramCube` and `SJICube` on `ndcube.NDCube`, the sequence on
      `NDCubeSequence`, the meta classes on `NDMeta` without the abstract bases.
- [ ] Port the four properties irispy uses, about forty lines: `spectral_axis`, `time`,
      `exposure_time`, `celestial`.
- [ ] Port the 27-line exposure-time correction as `apply_exposure_time_correction(cube)` beside
      `radiometric_calibration`, which is its only caller inside irispy.
- [ ] Then cut the sunraster-only metadata names: `instrument`, `observatory`,
      `sun_angular_radius`, `distance_to_sun`; keep `observer_radial_velocity`. 40 properties remain.

### 10. The dkist dependency

The slit-jaw reader imports `CoupledCompoundModel` and `VaryingCelestialTransform` from dkist
for per-frame pointing in the gwcs. They are about 160 lines on an 800-line base module, so
vendoring is not the answer.

- [ ] Open a gwcs issue proposing the varying celestial transform upstream; dkist stays meanwhile.

## Order of work

1. Phase 1 items in the order listed; each is a deletion-sized PR with existing tests.
2. Phase 2, items 9b then 12.
3. Phase 3, items 11 then 14.
4. Deferred items as upstream moves.

## Reference facts behind the decisions

| Measure | Value |
|---|---|
| Metadata properties defined / used by code, examples, docs | 81 / 12 |
| Public callables in `irispy.utils` before this plan | 41 |
| Helpers with no caller outside their module | 10 |
| sunraster members irispy uses | 5 |
| ndcube ASDF converter subclass support | none |
