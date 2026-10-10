# Retiring sunraster

Review written 2026-10-09 against sunraster 0.7.0 (main is 16 template commits ahead, with
PR #331 open), ndcube 2.4.2, sospice 0.0.9. It extends the roadmap in
[sunraster #284](https://github.com/sunpy/sunraster/issues/284) (2025-06-11) with the status
today and the ndcube side.

## Verdict

sunraster is 1,559 lines in four modules, and almost none of it belongs in ndcube. The #284
roadmap identifies every piece correctly, but routes two of them through sunpy core and through a
"gwcs of FITS WCSes" that nobody owns, which is why one item of seven is done sixteen months
later. Rerouting those two, deprecating the conveniences to calls ndcube already has, and letting
the instrument packages keep their own names makes the retirement a sequence of small,
independently mergeable steps. irispy is the only package on GitHub that imports sunraster (code
search for `from sunraster import`, 2026-10-09), so most of the work is ours.

## What sunraster holds and where each piece goes

| Feature | Lines | ndcube today | Destination | Status |
|---|---|---|---|---|
| `instrument_axes` | 7 | `NDMeta` with axis-aware keys | Stored in `NDMeta` since 0.7.0 (#273) | done |
| `spectral_axis`, `time`, `celestial` | 52, plus a 37-line name search over WCS, extra coords and meta | `axis_world_coords(physical_type, wcs=cube.combined_wcs)` | Recipes in the ndcube docs; instrument packages keep properties if they want them | not started |
| `exposure_time` | 12 | an axis-aware `NDMeta` entry | `meta["exposure time"]`, as irispy already stores it | not started |
| `apply_exposure_time_correction`, with `undo` and `force` | 27, plus 76 of helpers | `cube / NDData(exposure)` and `cube / Quantity` give the same unit and values (verified on an irispy cube) | Arithmetic, plus the small ndcube helper below | not started |
| `SpectrogramSequence` properties | in the 400-line sequence module | `NDCubeSequence.common_axis_coords` returns per-cube lists | One concatenation line, or the ndcube change below | not started |
| `RasterSequence` aliases, `slice_as_sns`, `slice_as_raster` | same module | Every alias maps one-to-one to an `NDCubeSequence` name (table in #284) | Deprecate to the ndcube names; `sns_instrument_axes_types` is derivable and goes | not started |
| Metadata ABCs (`MetaABC`, `RemoteSensorMetaABC`, `SlitSpectrographMetaABC`) | 108 | nothing | Drop. The roadmap sends them to sunpy core, where no issue exists and no one is asking | blocked by its own routing |
| SPICE reader and `SPICEMeta` (`instr/spice.py`) | 464 | nothing | [sospice PR 56](https://github.com/solo-spice/sospice/pull/56), open, last touched 2026-06-01, unblocked since 0.7.0 shipped on 2025-10-17 | stalled |

Two findings change the roadmap's shape:

- **The metadata ABCs do not need sunpy.** They enforce ten property names on two downstream
  meta classes. irispy and sospice can keep whatever names they choose by convention; irispy's
  are decided in `API_PLAN.md` item 1. Removing the ABCs from the critical path removes the one
  step that needs a community decision.
- **Sequence coordinates do not need a stacked WCS.** sunraster's `time` on a sequence
  concatenates per-cube extra coordinates (3 rasters x 8 steps gives a `Time` of 24), which
  ndcube's `common_axis_coords` already exposes as lists. A stacked gwcs would only matter for a
  sequence-level celestial grid, which sunraster computes by stacking anyway.

## What ndcube actually needs

Three small items, none of them a new class:

- [ ] **A broadcastable view of an axis-aware `NDMeta` entry**, e.g. `meta.as_array("exposure time")`
      returning the values reshaped to broadcast against the data. This is the only non-trivial
      code inside the exposure-time correction, about ten lines, and makes
      `cube / meta.as_array("exposure time")` the idiomatic replacement for the method. `NDMeta`
      already knows `axes` and `data_shape`.
- [ ] **`NDCubeSequence.common_axis_coords` returning coordinate objects**, a `Time`, `Quantity` or
      `SkyCoord` concatenated along the common axis, instead of nested lists. Small, and it is what
      every sunraster sequence property did.
- [ ] **A documentation page of recipes**: how a spectrograph cube reads its spectral axis, times,
      pointing and exposure times through `axis_world_coords` and `meta`, and how to divide by
      exposure time. Zero code; the deprecation warnings point at it.

Not needed for retirement: the `.coords` wrapper of ndcube #285 (dormant since 2020), a FITS-WCS
stacking wrapper, any instrument-axis vocabulary in ndcube. Those can come later on their own
merits. ndcube #730, a test-helper failure specific to sunraster, closes with the retirement.

## Retirement sequence

1. [ ] **sospice**: rebase PR 56 on sunraster 0.7.0, merge, release. The SPICE repositories that
       import `read_spice_l2_fits` (about eight on GitHub) switch imports. Independent of the
       steps below.
2. [ ] **irispy**: `API_PLAN.md` item 9a, moved up rather than deferred. `NDCube` bases for both
       cubes and `NDCubeSequence` for the sequence; the four properties (`spectral_axis`, `time`,
       `exposure_time`, `celestial`) ported as irispy's own, about forty lines; a ten-line
       `apply_exposure_time_correction(cube)` function beside `radiometric_calibration`, its only
       caller; meta without the ABCs, cutting `instrument`, `observatory`, `sun_angular_radius`
       and `distance_to_sun`. irispy then drops the dependency. Independent of step 1.
3. [ ] **ndcube**: the two small PRs and the recipes page, released in the next minor.
4. [ ] **sunraster 0.8**, a deprecation release: every property, the correction method, the
       `RasterSequence` aliases and the ABCs warn and name their replacement from step 3;
       `instr.spice` warns and points at sospice. One release cycle of grace.
5. [ ] **sunraster final**: an end-of-life release whose changelog is the replacement table above;
       archive the repository; mark the sunpy affiliated listing and the conda-forge feedstock.
       Only after steps 1 and 2 and the grace period.

Steps 1 to 3 can run in parallel; step 4 needs step 3 for the warning text; step 5 waits on
everything. PR #331 on sunraster, a `__str__` change, is not worth merging into a package on this
path.

## Not checked

The conda-forge feedstock and the affiliated-package listing for sunraster, and the diff of
sospice PR 56. The irispy step changes base classes, so it carries the test churn; it should
land before the sunraster deprecation release so irispy users never see the warnings.
