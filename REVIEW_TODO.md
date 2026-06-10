# gWCS raster branch — review findings & fixes

From the multi-angle review of `gwcs_raster` vs `main` (2026-06-10).
Ordered by severity within each section. Items 1–2 first: they corrupt every sky
coordinate and mask everything downstream that compares against `basic_wcs`.

## Correctness — coordinate core

### 1. gWCS celestial lon/lat are swapped for every raster cube
- **Where:** `irispy/_spectrograph_wcs.py:435-478` (`_create_raster_gwcs`)
- **What:** The `VaryingCelestialTransform` output is already (lon, lat)-ordered; the
  appended `Mapping((1,0))` "SwapHelioprojectiveAxes" inverts it. The no-swap
  sit-and-stare branch feeds the lat-axis cdelt/pc into the projection x input.
- **Evidence:** Verified by execution on the bundled 13-file raster:
  `cube.wcs.array_index_to_world(0,50,3)` → Tx=272.19/Ty=481.02 while `basic_wcs`
  gives Tx=481.77/Ty=279.61 (209.58″ Tx error). The branch's own
  `test_raster_gwcs_matches_basic_wcs_forward_world_coordinates` fails. For
  sit-and-stare, the along-slit displacement appears on Tx instead of Ty.
- **Fix:** Remove the inverting `Mapping((1,0))` (and its inverse counterpart) from
  the raster branch; in the sit-and-stare branch route the lon-axis cdelt/pc
  components to the projection x input. Validate `array_index_to_world` against
  `basic_wcs` at several pixels for raster, sit-and-stare, and v34 data.

### 2. CRPIX passed 1-based to VaryingCelestialTransform → one-pixel sky offset
- **Where:** `irispy/_spectrograph_wcs.py:420`
- **What:** `crpix_table` passes raw FITS CRPIX2/CRPIX3 (1-based) to dkist's
  `VaryingCelestialTransform`, whose `Shift(-crpix)` operates on 0-based gWCS pixel
  inputs. The spectral axis a few lines above correctly subtracts `crpix1 - 1`.
- **Evidence:** Verified numerically: transform built with `crpix_table=[3,4]` puts
  crval at 0-based pixel (3,4); the equivalent astropy WCS puts it at (2,3). Every
  sky position is offset ~0.17″ along the slit and one raster step along scan;
  `crop()`/`world_to_pixel` select neighbouring pixels.
- **Fix:** Subtract 1 from CRPIX2/CRPIX3 before building `crpix_table`. Add a
  regression test asserting gWCS and astropy WCS agree exactly at the reference
  pixel. Land with item 1, then re-run the forward-coordinates test.

### 3. Slicing leaves stale per-step pc/crval tables (step axis ignored)
- **Status:** Fixed by slicing sidecar WCS tables with both scan and step items.
- **Where:** `irispy/_spectrograph_wcs.py:242` (`_slice_raster_metadata`,
  separate-raster-axis branch)
- **What:** Tables are sliced only by the scan index (`value[scan_item]`); the
  step-axis component of the slice item is dropped.
- **Evidence:** Verified: `cube[:, 0:4]` on the 13-file cube → data (13,4,109,29)
  but `_raster_pc_table` stays (13,8,2,2). Recombining `split_rasters()` builds a
  gWCS with 8-step lookup tables for 4-step data.
- **Fix:** Apply the step-axis component of the normalized item to axis 1 of both
  tables (`value[scan_item][:, step_item]`, with scalar handling consistent with
  data slicing). Test: slice, assert table shapes track data; round-trip
  `split_rasters()` → `_build_combined_raster_cube` produces matching coordinates.

### 4. Sit-and-stare PC-only-zero AUX rows are never flagged
- **Status:** Fixed with a sit-and-stare-only PC bad-row mask path.
- **Where:** `irispy/_spectrograph_wcs.py:281-291` (`_raster_wcs_bad_row_mask`)
- **What:** Mask requires `pc_bad & crval_bad`. The sit-and-stare path fills crval
  from header CRVAL3/CRVAL2 (nonzero off disk-center), so an unfilled AUX row with
  an all-zero PC matrix is never flagged — contradicting the reader comment at
  `io/spectrograph.py:193` ("only PC may need fallback"). That exposure's sky
  coordinates silently collapse to crval.
- **Fix:** Add a `pc_only=False` parameter; the sit-and-stare branch in
  `read_spectrograph_lvl2` passes `pc_only=True` so all-zero-PC rows are flagged
  regardless of crval. Keep the AND rule for the raster path (crval=(0,0) is valid
  at disk center). Unit test: synthetic sit-and-stare AUX table with one zero PC
  row → row gets interpolated.

### 5. Sit-and-stare detection via `np.isclose` sentinel is fragile (three variants)
- **Status:** Fixed by deciding sit-and-stare exactly in the reader and passing it through WCS construction.
- **Where:** `irispy/_spectrograph_wcs.py:435` & `:472` (`np.isclose(CDELT3, 1e-10)`
  hardcoded, ignoring the defined `SIT_AND_STARE_CDELT3_PLACEHOLDER`),
  `irispy/io/spectrograph.py:192` (`np.isclose(CDELT3, 0)`)
- **What:** `np.isclose` default `atol=1e-8` classifies any |CDELT3| < ~1e-8 as
  sit-and-stare, so a legitimately tiny step size silently takes the wrong
  transform path (no axis swap, wrong PC handling). Changing the placeholder
  constant also silently breaks detection because two checks use the literal.
- **Fix:** Decide the mode once in the reader (`window_header["CDELT3"] == 0`
  exact, before placeholder substitution) and pass an explicit
  `sit_and_stare: bool` through `_prepare_raster_wcs_header` and
  `_create_raster_gwcs`. Use `SIT_AND_STARE_CDELT3_PLACEHOLDER` wherever the
  placeholder value is still needed; delete the literal `1e-10` comparisons.

## Correctness — API / pipeline breakage

### 6. Combined multi-file cubes drop the per-step `time` extra coord
- **Where:** `irispy/io/_raster_combine.py:168` (`_build_combined_raster_cube`),
  same for the lazy combiner
- **What:** Combined cubes are constructed without `extra_coords`, dropping the
  `time` coordinate single-file cubes carry. The documented pattern
  `cube.axis_world_coords("time", wcs=cube.extra_coords)` (used in
  `examples/coalign/03_offset_sji_sg.py:59`, `examples/how_to/03_work_with_rasters.py`)
  crashes on any multi-file read.
- **Fix:** Collect each cube's `extra_coords["time"]`, stack to a
  (n_scans, n_steps) Time array, add to the combined cube's extra_coords on axes
  (0, 1). Mirror in `_combine_raster_cubes_lazy`. Test: multi-file read, assert
  `axis_world_coords("time", wcs=cube.extra_coords)` returns (n_scans, n_steps).

### 7. `remove_cosmic_rays` breaks on dask-backed (memmap multi-file) cubes
- **Where:** `irispy/utils/cosmic_rays.py:119`
- **What:** `np.asarray(cube.mask)` forces a full compute of the whole observation
  (mask is a separate dask graph), then the backends pass dask 2D slices to
  astroscrappy's Cython `detect_cosmics` → TypeError. The lazy-read design is
  defeated and the method crashes.
- **Fix:** Detect dask-backed data and materialize per frame inside the loop
  (`np.asarray(data[index])`, `np.asarray(mask[index])`), never the full cube. If
  a backend genuinely needs the full cube, raise a clear ValueError advising to
  slice or load without memmap.

### 8. Double exposure-time correction (removed sunraster guard)
- **Where:** `irispy/utils/spectrograph.py:122` (`radiometric_calibration`)
- **What:** `cube.apply_exposure_time_correction()` was replaced by unconditional
  `cube / _reshape_exposure_time_for_broadcast(cube)`. sunraster's method only
  applied the correction when the unit did not already contain inverse time. A
  cube already in DN/s is divided twice: unit becomes DN/s², `unit.to(u.photon/u.s)`
  raises UnitConversionError, and the data would be wrong by 1/t_exp regardless.
- **Fix:** Restore the guard: skip division when `u.s` appears with negative power
  in `cube.unit.decompose().bases/powers`. Better altitude: override
  `apply_exposure_time_correction` on irispy's `SpectrogramCube` to handle the 4D
  combined cube and call that, instead of bypassing it while coupling to three
  sunraster privates (`_get_axis_coord_index`, `_exposure_time_name`,
  `_exposure_time_loc`).

### 9. Raster-scanning combine contract mismatch: 4D stack vs 3D concat
- **Where:** `irispy/io/tests/test_utils.py:132` vs
  `irispy/io/_raster_combine.py` (`_combine_raster_cubes*`)
- **What:** `test_read_files_raster_scanning` asserts the combined C II cube is 3D
  (116, 388, 186) (29 rasters × 4 steps merged on one axis), but the combiner
  always stacks on a new axis 0 → (29, 4, 388, 186). The concatenate path the test
  implies exists only as the dead `_concatenate_*` helper family.
- **Fix:** Decide the contract. If scanning rasters should concatenate along the
  step axis (3D, flat scan axis with `_raster_boundaries` recording (start, stop)
  per file), wire the `_concatenate_*` family into `_finalize_window_object` for
  that case. Otherwise fix the test to expect 4D and update
  `aligned_dimensions` expectations. Either way, delete whichever helper family
  ends up unused (see item 15).

### 10. Ragged final raster makes the whole observation unloadable
- **Where:** `irispy/io/_raster_combine.py:140` (`_validate_combinable_raster_cubes`)
- **What:** Hard ValueError on any shape mismatch. The replaced
  `SpectrogramCubeSequence(common_axis=0)` tolerated ragged scan lengths, so
  observations aborted mid-scan (last file has fewer exposures) loaded fine before
  and now fail entirely for every spectral window.
- **Fix:** Pad the short last cube with NaN data + True mask up to the common step
  count before stacking (preserves data, keeps the 4D shape, mask marks missing
  steps); also pad uncertainty/pc/crval tables (repeat last good row) and emit a
  UserWarning. Test with a truncated final file.

### 11. Removed warn-and-skip guard around per-window WCS construction
- **Where:** `irispy/io/spectrograph.py:209` (`basic_wcs = WCS(prepared_wcs_header)`)
- **What:** Old reader wrapped `WCS(header)` in try/except → logged a warning and
  skipped that cube. Now a single corrupt window header aborts the entire
  (potentially multi-GB, multi-file) read.
- **Fix:** Restore try/except around `WCS(prepared_wcs_header)` +
  `_create_raster_gwcs` per window: log filename/window/step, skip that cube,
  continue. Ensure `_finalize_window_object` handles a window with fewer cubes
  than files (or zero cubes → drop the window with a warning).

### 12. Two tests fail deterministically
- **Status:** Fixed with a test-local `pytest.mark.filterwarnings`.
- **Where:** `irispy/io/tests/test_spectrograph.py:182`
  (`test_memmap_mode_never_computes_uncertainty`)
- **What:** Calls `read_spectrograph_lvl2(memmap=True, uncertainty=True)` without
  `pytest.warns`; the reader now emits a UserWarning for that combination and
  pytest config sets `filterwarnings = error`. Confirmed failing locally.
- **Fix:** Filter the expected warning on this test with
  `pytest.mark.filterwarnings`.
  (The other deterministic failure is the shape assertion in item 9.)

### 13. Stale 2-element crop calls in v34 example
- **Where:** `examples/how_to/04_v34_rasters.py:107-114`
- **What:** Two `crop()` calls still pass `[SpectralCoord, target]` (2 world
  objects) against the new 4-world-axis gWCS (SpectralCoord, SkyCoord, Time,
  raster-step Quantity). astropy's `high_level_objects_to_values` raises
  ValueError; the docs gallery build crashes. The crop at line 71 of the same
  file was already updated to 4 entries.
- **Fix:** Update both calls to
  `[SpectralCoord(...), target, None, None]`.

## Cleanup / design

### 14. 13 private sidecar attributes manually re-threaded everywhere
- **Where:** `irispy/_spectrograph_wcs.py:95` (`_SPECTROGRAM_CUBE_METADATA_KWARGS`),
  call sites in `utils/dust.py:192`, `utils/cosmic_rays.py:152`,
  `utils/spectrograph.py:148`
- **Cost:** Any operation that produces a derived cube and forgets
  `_spectrogram_cube_metadata_kwargs_for_copy` (arithmetic, rebin, any future
  utility) silently drops all raster metadata: `basic_wcs` → None, raster
  slicing/calibration degrade with no error.
- **Fix:** Override `_new_instance`/`to_nddata` on `SpectrogramCube` to copy the
  sidecar attrs automatically; delete the helper threading at call sites.

### 15. Dead `_concatenate_*` helper family (~80 lines)
- **Where:** `irispy/io/_raster_combine.py:18-100`
  (`_concatenate_scan_aligned_values`, `_concatenate_uncertainty`,
  `_concatenate_mask`, `_combine_raster_meta`)
- **Cost:** Never called; near-verbatim copies of the `_stack_*` family. Fixes to
  one family silently miss the other.
- **Fix:** Resolve item 9 first; then either wire these in for scanning rasters or
  delete them (or fold each pair into one helper taking np.stack/np.concatenate).

### 16. AUX slit-offset columns re-hardcoded as magic numbers
- **Where:** `irispy/io/spectrograph.py:203`
  (`offset_index = 34 if meta.spectral_band == "FUV" else 45`)
- **Cost:** Duplicates `AUX_FUV_SLIT_OFFSET_COLUMN`/`AUX_NUV_SLIT_OFFSET_COLUMN`
  already defined in `_spectrograph_wcs.py:26-27` and used by
  `_prepare_raster_wcs_header` for the identical correction. Divergence silently
  mis-points gWCS vs basic_wcs by half a slit width.
- **Fix:** Import the constants (or add `_slit_offset_column(spectral_band)`) and
  use them in both places.

### 17. Combiner dispatch duplication + unreachable guard
- **Where:** `irispy/io/_raster_combine.py:250-263`
- **Cost:** `_finalize_window_object` → `_combine_raster_cubes` /
  `_combine_raster_cubes_lazy` each repeat validate + len==1 early-return; the
  eager path's memmap NotImplementedError is unreachable from the public reader;
  `create_raster_gwcs` is threaded as a parameter through four functions but only
  ever bound to `_create_raster_gwcs` (importable directly, no cycle). The two
  paths already have divergent mask semantics.
- **Fix:** One `_combine_raster_cubes(cubes, *, memmap)` that imports
  `_create_raster_gwcs` and picks lazy vs `np.stack` internally; single
  validation; drop the dead guard.

### 18. Reuse: replace hand-rolled implementations
- `_normalize_basic_wcs_item` (`_spectrograph_wcs.py:107`) re-implements slice
  normalization — use astropy's `sanitize_slices` (already imported at line 12).
- `_interpolate_wcs_bad_rows` (`_spectrograph_wcs.py:294`) hand-rolls per-row
  linear interpolation — use `np.interp` like `io/sji.py:117-121` does for the
  same all-zero-AUX-row problem; extract a shared helper for both readers.
- `_header_time` (`io/spectrograph.py:32`) uses strict
  `Time(value, format="isot")` — use `sunpy.time.parse_time` (already used in
  `_create_raster_gwcs`) for consistent tolerant parsing.
- `_slice_basic_wcs` (`_spectrograph_wcs.py:151-171`) has two near-duplicate
  segment-lookup loops — merge into one with a conditional prefix.
- `_raster_boundaries` stored on separate-axis cubes is dead state (the property
  re-derives from shape) — stop passing it for that case.

### 19. Efficiency
- **Eager combine 2× peak memory** (`_raster_combine.py:257`): `np.stack` (data,
  uncertainty, mask) while source cubes stay referenced. Preallocate with
  `np.empty`, copy per cube, drop each cube's arrays as consumed.
- **Per-file gWCS built then discarded** (`io/spectrograph.py:249`): when
  len(filenames) > 1, every file×window builds a full gWCS that the combiner
  throws away. Skip `_create_raster_gwcs` in the loop (pass tables through) and
  build it once per window in `_finalize_window_object`.
- **4 MB dask chunks → fits.open per ~8 steps**
  (`LAZY_RASTER_CHUNK_TARGET_BYTES`, `_raster_combine.py:15`): raise to 64–256 MB
  or one chunk per file; header parsing then happens once per file.
- **Mask as separate dask graph** (`_raster_combine.py:246`): computing data and
  mask separately re-reads all files twice. Build both in one `map_blocks` /
  shared compute, or leave mask None until materialization.
- **Per-frame axis-property rebuild** (`visualization.py:199`):
  `update_plot` re-runs `set_axis_properties(animate=True)` every slider tick,
  forcing full tick re-layout + gWCS boundary evaluation per frame. Cache the
  last-applied WCS/slices identity and re-apply only when the visible WCS changes.

### 20. Plotting selects pseudo-axis coords by display-label string matching
- **Where:** `irispy/visualization.py:143` (`SCAN_STEP_LABELS`,
  `SLIDER_SCAN_STEP_LABELS`, `_shorten_slider_label` splitting on " / ")
- **Cost:** Renaming an axes_name in the gWCS frame or an upstream change to
  WCSAxes/mpl_animators label formatting silently breaks the hiding logic — ghost
  ticks reappear, sliders show raw labels, only visible by eye.
- **Fix:** Give the step/scan frames real `axis_physical_types` in
  `_create_raster_gwcs` and select/hide coords by physical type (or configure via
  `coord_params` once in `IRISPlotter`), not display labels.

### 21. Meta combining patches privates with duplicated key lists
- **Where:** `irispy/io/_raster_combine.py:103`
  (`_combine_raster_sequence_meta`)
- **Cost:** Reaches into `meta._data_shape`, hand-edits NAXIS3/NAXIS4 in meta and
  fits_header, and re-stacks a hardcoded key tuple ('exposure time', 'exposure FOV
  center', 'observer radial velocity', 'orbital phase') that must mirror the
  `meta.add` calls in `io/spectrograph.py:182-185`. A new per-exposure key added
  in the reader silently doesn't combine.
- **Fix:** Add `SGMeta.combine(metas, new_shape)` classmethod that stacks all
  axis-0-aligned entries generically (SGMeta records axis alignment), and use it
  from the combiner.
