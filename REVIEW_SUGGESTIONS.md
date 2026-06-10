# gWCS raster branch — general suggestions

Branch-level recommendations from the 2026-06-10 review, beyond the specific
findings in `REVIEW_TODO.md`. Nothing here is blocking; the coordinate fixes
(REVIEW_TODO items 1–2) remain the critical path.

## Packaging

- **Declare `gwcs` explicitly.** `_spectrograph_wcs.py:9` imports `gwcs`
  directly, but pyproject only gets it transitively via `dkist`. Transitive
  dependencies can be dropped or change majors without warning; add
  `gwcs>=<minimum you actually need>` to `dependencies`.
- **Trim `dask[distributed]` to `dask[array]`.** The combiner only uses
  `dask.array` and `dask.delayed`; the `distributed` extra pulls in tornado,
  zict, and friends for nothing.
- **Reconsider the dkist dependency weight.** `dkist>=1.17.0` is a heavy
  dependency for essentially one transform family
  (`VaryingCelestialTransform`). Worth raising with the dkist devs whether
  those models could live in gwcs or a small shared package — pairs naturally
  with the upstreaming conversation below.

## Test strategy

- **Make the gWCS-vs-basic_wcs consistency test the gate for the branch.** It
  currently fails on the bundled data and is exactly the test that catches the
  core bugs. Extend it to a parametrized matrix:
  {raster multi-file, single-file, sit-and-stare, v34, v34 + revert_v34} ×
  {forward world coords, world_to_pixel round-trip}.
- **Add round-trip tests.** `world_to_pixel(pixel_to_world(idx)) ≈ idx` would
  have caught both the lon/lat swap and the CRPIX off-by-one *independently*
  of basic_wcs — it requires no reference WCS to compare against.
- **Don't leave scanning-raster coverage remote-only.** The 4D-vs-3D combine
  contract mismatch sat unnoticed because the only test needs `--remote-data`.
  Run remote-data tests on a CI cron at minimum, or commit a tiny synthetic
  2-scan × 2-step fixture so the contract is exercised on every push.
- **Keep `filterwarnings = error`.** It surfaced a real bug during this
  review. Fix tests with `pytest.warns(...)` wrappers rather than relaxing the
  config.
- **Lock in the lazy path with one integration test.** Read a memmap
  multi-file observation, compute a single slice, and assert the number of
  `fits.open` calls via mock. This pins the chunking behaviour once
  REVIEW_TODO item 19 is fixed.

## Branch hygiene

- **Gitignore the scratch data at repo root.**
  `iris_l2_20140329_140938_3860258481_raster/`,
  `iris_l2_20240509_170727_4204700143_raster_t000_r00000.fits`, and `AGENTS.md`
  are untracked; one accidental `git add .` puts FITS files in history
  permanently.
- **Squash the WIP history before PR.** "WIP", "WIP Fixes", "Broke deps" →
  reorganize into logical commits (new WCS module / combiner / reader rewrite /
  visualization / examples & docs) so reviewers can read and bisect.

## API & migration

- **Soften the `crop()` signature break.** Going from 2 to 4 world objects is
  the biggest silent user break: old 2-element calls die with an opaque astropy
  ValueError deep in `high_level_objects_to_values`. Catch that in
  `SpectrogramCube.crop` and re-raise naming the new 4-axis world order
  (SpectralCoord, SkyCoord, Time, raster step). Cheap, and saves every
  downstream user a debugging session.
- **Write a migration guide section in the docs.** Changelog entries exist,
  but show before/after for the common patterns: window indexing
  (`raster["w"][0]` → `raster["w"]`), crop points, celestial frame access,
  time access, per-raster access (`raster_slice` / `split_rasters`).
- **Document (or hide) the `wcs` vs `basic_wcs` split.** Examples now mix the
  two (`wcs_to_celestial_frame(cube.basic_wcs.celestial)`) — a sign the gWCS
  isn't drop-in. Either document clearly when each applies, or add a
  `celestial`-like convenience on the cube so users never need `basic_wcs` for
  the common case. Naming: `fits_wcs` communicates what it is better than
  `basic_wcs`.
- **Align the OBS-consistency claim.** The reader docstring says same-OBS is
  "not checked by this function by design", but the combiner does check OBSID.
  Make the docs match reality.

## Design / upstream

- **Open ndcube/sunraster issues before polishing the workaround machinery.**
  The 13 sidecar attributes, `_basic_wcs_segments`, and the custom slicing
  logic are irispy fighting upstream limitations (gWCS-aware slicing, per-axis
  lookup-table WCS, metadata propagation through derived cubes via
  `_new_instance`). Some of that likely belongs upstream; upstreaming shrinks
  this branch's permanent maintenance surface.
- **Consider time as the step axis's primary world coordinate.** The pseudo
  axis currently exposes a dimensionless step Quantity, with time bolted on as
  an extra coord. Users ask "when was this exposure" far more often than
  "which step index" — making time the primary coordinate of that axis would
  also remove the dropped-extra-coord failure mode (REVIEW_TODO item 6) by
  construction.
